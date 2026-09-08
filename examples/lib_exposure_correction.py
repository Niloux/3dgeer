# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PPISP formulas adapted from nv-tlabs/ppisp at
# df33809f7b3b20ac06de088dfc871b144b8fb54d (ppisp_math.cuh and torch_reference.py).
# Modified for batched local parameters, nonnegative radiance, and a frozen
# identity CRF. Global/local decomposition follows LichtFeld's exposure mode.
# Licensed under the Apache License, Version 2.0:
# https://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

"""PyTorch PPISP exposure compensation with a local exposure/chroma grid.

The local path intentionally uses autograd/grid_sample, not fused-bilagrid's
12-channel affine kernels. No external PPISP package or CUDA build is needed.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ColorCorrection(nn.Module):
    """PPISP chromaticity homography for global or per-pixel latent parameters."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "pinv",
            torch.tensor(
                [
                    [[0.0480542, -0.0043631], [-0.0043631, 0.0481283]],
                    [[0.0580570, -0.0179872], [-0.0179872, 0.0431061]],
                    [[0.0433336, -0.0180537], [-0.0180537, 0.0580500]],
                    [[0.0128369, -0.0034654], [-0.0034654, 0.0128158]],
                ]
            ),
        )

    def forward(self, rgb: Tensor, params: Tensor) -> Tensor:
        params = torch.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        offsets = torch.einsum(
            "...ki,kij->...kj", params.reshape(*params.shape[:-1], 4, 2), self.pinv
        )
        bx, by = offsets[..., 0, :].unbind(-1)
        rx, ry = offsets[..., 1, :].unbind(-1)
        gx, gy = offsets[..., 2, :].unbind(-1)
        nx, ny = offsets[..., 3, :].unbind(-1)
        rx, gy, nx, ny = rx + 1, gy + 1, nx + 1 / 3, ny + 1 / 3

        # Rows of skew(t_neutral) @ [t_blue, t_red, t_green].
        tx = torch.stack((bx, rx, gx), -1)
        ty = torch.stack((by, ry, gy), -1)
        row0 = ny.unsqueeze(-1) - ty
        row1 = tx - nx.unsqueeze(-1)
        row2 = nx.unsqueeze(-1) * ty - ny.unsqueeze(-1) * tx
        lam01 = torch.linalg.cross(row0, row1, dim=-1)
        lam02 = torch.linalg.cross(row0, row2, dim=-1)
        lam12 = torch.linalg.cross(row1, row2, dim=-1)
        lam = torch.where(
            lam01.square().sum(-1, keepdim=True) >= 1e-20,
            lam01,
            torch.where(
                lam02.square().sum(-1, keepdim=True) >= 1e-20, lam02, lam12
            ),
        )
        # H = T @ diag(lam) @ S_inv, H[2,2] = lam_blue. Apply directly
        # in RGB order to avoid materializing a 3x3 matrix at every pixel.
        scale = lam[..., :1]
        lam = lam / torch.where(scale.abs() > 1e-20, scale, torch.ones_like(scale))
        rgb = F.relu(rgb)  # zero subgradient for nonpositive radiance
        weighted = rgb[..., [2, 0, 1]] * lam
        red = (weighted * tx).sum(-1)
        green = (weighted * ty).sum(-1)
        intensity_out = weighted.sum(-1)
        norm = rgb.sum(-1) / (F.relu(intensity_out) + 1e-5)
        return (
            torch.stack((red, green, intensity_out - red - green), -1)
            * norm.unsqueeze(-1)
        )


def apply_exposure(rgb: Tensor, exposure: Tensor) -> Tensor:
    exposure = torch.nan_to_num(exposure, nan=0.0, posinf=0.0, neginf=0.0)
    return rgb * torch.exp2(exposure.clamp(-16.0, 16.0)).unsqueeze(-1)


def photometric_scheduler(optimizer, max_steps: int, warmup_steps: int):
    """Linear warmup then exponential decay, also defined for short runs."""
    max_steps = max(max_steps, 1)
    warmup_steps = min(warmup_steps, max_steps - 1)

    def factor(step):
        if step < warmup_steps:
            return 0.01 + 0.99 * step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.01 ** min(progress, 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class PPISPExposure(nn.Module):
    """Per-frame exposure/color and per-physical-camera vignetting.

    The CRF is fixed to identity including its [0,1] input clamp. Novel views
    (frame_idx=-1) use only the shared camera vignetting, with no GT fitting.
    """

    def __init__(self, num_cameras: int, num_frames: int, lr: float = 2e-3):
        super().__init__()
        self.exposure_params = nn.Parameter(torch.zeros(num_frames))
        self.color_params = nn.Parameter(torch.zeros(num_frames, 8))
        self.vignetting_params = nn.Parameter(torch.zeros(num_cameras, 3, 5))
        self.color_correction = ColorCorrection()
        self.lr = lr

    def forward(self, rgb: Tensor, camera_idx: int, frame_idx: int = -1) -> Tensor:
        height, width = rgb.shape[:2]
        if frame_idx >= 0:
            rgb = apply_exposure(rgb, self.exposure_params[frame_idx])
        vig = self.vignetting_params[camera_idx]
        u = (
            torch.arange(width, device=rgb.device, dtype=rgb.dtype) + 0.5 - width / 2
        ) / max(width, height)
        v = (
            torch.arange(height, device=rgb.device, dtype=rgb.dtype) + 0.5 - height / 2
        ) / max(width, height)
        r2 = (u[None, :, None] - vig[:, 0]).square() + (
            v[:, None, None] - vig[:, 1]
        ).square()
        falloff = 1 + r2 * (vig[:, 2] + r2 * (vig[:, 3] + r2 * vig[:, 4]))
        rgb = rgb * falloff.clamp(0.0, 1.0)
        if frame_idx >= 0:
            rgb = self.color_correction(rgb, self.color_params[frame_idx])
        return rgb.clamp(0.0, 1.0)

    def get_regularization_loss(self) -> Tensor:
        vig = self.vignetting_params
        return (
            0.02 * vig[..., :2].square().sum(-1).mean()
            + 0.01 * F.relu(vig[..., 2:]).mean()
            + 0.1 * vig.var(dim=1, unbiased=False).mean()
        )

    @torch.no_grad()
    def project_mean(self):
        self.exposure_params.sub_(self.exposure_params.mean())
        self.color_params.sub_(self.color_params.mean(dim=0, keepdim=True))

    def create_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-15)]

    def create_schedulers(self, optimizers, max_steps):
        return [
            photometric_scheduler(optimizer, max_steps, 500)
            for optimizer in optimizers
        ]


class ExposureChromaGrid(nn.Module):
    """Per-image (x,y,luminance) grid of exposure + eight PPISP color latents."""

    def __init__(self, num_images: int, shape=(16, 16, 8)):
        super().__init__()
        width, height, guidance = shape
        self.grids = nn.Parameter(torch.zeros(num_images, 9, guidance, height, width))
        self.color_correction = ColorCorrection()
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]))

    def forward(self, rgb: Tensor, image_ids: Tensor) -> Tensor:
        batch, height, width, _ = rgb.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=rgb.device, dtype=rgb.dtype),
            torch.linspace(-1, 1, width, device=rgb.device, dtype=rgb.dtype),
            indexing="ij",
        )
        # Guidance uses the PPISP-corrected image and remains differentiable.
        z = (rgb * self.luma).sum(-1).clamp(0.0, 1.0) * 2 - 1
        coords = torch.stack(
            (x.expand(batch, -1, -1), y.expand(batch, -1, -1), z), -1
        )
        sampled = F.grid_sample(
            self.grids[image_ids.reshape(-1)],
            coords.unsqueeze(1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2).permute(0, 2, 3, 1)
        return self.color_correction(
            apply_exposure(rgb, sampled[..., 0]), sampled[..., 1:]
        )

    def tv_loss(self) -> Tensor:
        # Same squared-TV normalization as lib_bilagrid, for all nine channels.
        return sum(self.grids.diff(dim=axis).square().mean() for axis in (2, 3, 4))

    @torch.no_grad()
    def project_mean(self):
        self.grids.sub_(self.grids.mean(dim=(2, 3, 4), keepdim=True))
