import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps


def so3_exp_map(omega: Tensor) -> Tensor:
    """Convert an axis-angle vector to a rotation matrix."""
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta = theta2.clamp_min(1e-12).sqrt()
    theta4 = theta2 * theta2
    small = theta2 < 1e-8
    A = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta4 / 120.0,
        torch.sin(theta) / theta,
    )
    B = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta4 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )

    x, y, z = omega.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*omega.shape[:-1], 3, 3)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype)
    eye = eye.expand(*omega.shape[:-1], 3, 3)
    return eye + A[..., None] * skew + B[..., None] * torch.matmul(skew, skew)


def so3_log_map(rotation: Tensor) -> Tensor:
    """Convert rotation matrices to axis-angle vectors."""
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    theta = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    scale = torch.where(
        theta < 1e-5,
        0.5 + theta.square() / 12.0,
        theta / (2.0 * sin_theta).clamp_min(1e-8),
    )
    return vee * scale[..., None]


class _PoseDeltaModule(torch.nn.Module):
    """Shared implementation for per-image and per-rig pose corrections."""

    def __init__(self, n: int, reference_index: int | None, rotation_mode: str):
        super().__init__()
        if n <= 0:
            raise ValueError("Pose optimization requires at least one pose")
        if rotation_mode not in {"so3", "6d"}:
            raise ValueError("rotation_mode must be 'so3' or '6d'")
        if reference_index is not None and not 0 <= reference_index < n:
            raise ValueError("reference_index is outside the pose table")
        self.rotation_mode = rotation_mode
        self.reference_index = reference_index
        self.trans = torch.nn.Embedding(n, 3, padding_idx=reference_index)
        self.rot = torch.nn.Embedding(
            n, 3 if rotation_mode == "so3" else 6, padding_idx=reference_index
        )
        self.register_buffer(
            "identity",
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            persistent=False,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate checkpoints written by the old 9D pose embedding."""
        legacy_key = prefix + "embeds.weight"
        trans_key = prefix + "trans.weight"
        rot_key = prefix + "rot.weight"
        if legacy_key in state_dict and trans_key not in state_dict:
            legacy = state_dict.pop(legacy_key)
            state_dict[trans_key] = legacy[..., :3]
            legacy_rot = legacy[..., 3:]
            if self.rotation_mode == "so3":
                identity = self.identity.to(legacy_rot).expand_as(legacy_rot)
                state_dict[rot_key] = so3_log_map(
                    rotation_6d_to_matrix(legacy_rot + identity)
                )
            else:
                state_dict[rot_key] = legacy_rot
        state_dict.pop(prefix + "identity", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._zero_reference()

    @property
    def embeds(self):
        """Compatibility view for old callers that only inspect `.embeds.weight.grad`."""
        return _LegacyPoseEmbeddingView(self)

    def zero_init(self):
        torch.nn.init.zeros_(self.trans.weight)
        torch.nn.init.zeros_(self.rot.weight)
        self._zero_reference()

    def random_init(self, std: float):
        torch.nn.init.normal_(self.trans.weight, std=std)
        torch.nn.init.normal_(self.rot.weight, std=std)
        self._zero_reference()

    def _zero_reference(self):
        if self.reference_index is not None:
            with torch.no_grad():
                self.trans.weight[self.reference_index].zero_()
                self.rot.weight[self.reference_index].zero_()

    def _delta_transform(self, pose_ids: Tensor) -> Tensor:
        batch_dims = pose_ids.shape
        dx = self.trans(pose_ids)
        drot = self.rot(pose_ids)
        if self.rotation_mode == "so3":
            rot = so3_exp_map(drot)
        else:
            rot = rotation_6d_to_matrix(
                drot + self.identity.to(drot).expand(*batch_dims, -1)
            )
        transform = torch.eye(4, device=dx.device, dtype=dx.dtype)
        transform = transform.expand(*batch_dims, 4, 4).clone()
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return transform

    def prior_loss(self, translation_sigma: float, rotation_sigma: float) -> Tensor:
        """Normalized physical prior for all non-reference pose corrections."""
        trans = self.trans.weight
        rot = self.rot.weight
        if self.reference_index is not None:
            mask = torch.ones(trans.shape[0], dtype=torch.bool, device=trans.device)
            mask[self.reference_index] = False
            trans, rot = trans[mask], rot[mask]
        if trans.numel() == 0:
            return self.trans.weight.sum() * 0.0
        return (
            trans.square().sum(-1) / max(translation_sigma, 1e-8) ** 2
            + rot.square().sum(-1) / max(rotation_sigma, 1e-8) ** 2
        ).mean()

    @torch.no_grad()
    def metrics(self) -> dict[str, Tensor]:
        trans_norm = torch.linalg.vector_norm(self.trans.weight, dim=-1)
        if self.rotation_mode == "so3":
            rot_norm = torch.linalg.vector_norm(self.rot.weight, dim=-1)
        else:
            rot_matrix = rotation_6d_to_matrix(
                self.rot.weight + self.identity.to(self.rot.weight)
            )
            cosine = (
                (rot_matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
            ).clamp(-1.0, 1.0)
            rot_norm = torch.acos(cosine)
        if self.reference_index is not None:
            mask = torch.ones_like(trans_norm, dtype=torch.bool)
            mask[self.reference_index] = False
            trans_norm, rot_norm = trans_norm[mask], rot_norm[mask]
        if trans_norm.numel() == 0:
            zero = self.trans.weight.new_zeros(())
            return {"translation_mean": zero, "translation_p95": zero,
                    "translation_max": zero, "rotation_mean_deg": zero,
                    "rotation_p95_deg": zero, "rotation_max_deg": zero}
        rotation_deg = rot_norm * (180.0 / math.pi)
        return {
            "translation_mean": trans_norm.mean(),
            "translation_p95": torch.quantile(trans_norm, 0.95),
            "translation_max": trans_norm.max(),
            "rotation_mean_deg": rotation_deg.mean(),
            "rotation_p95_deg": torch.quantile(rotation_deg, 0.95),
            "rotation_max_deg": rotation_deg.max(),
        }


class _LegacyPoseEmbeddingView:
    def __init__(self, module: _PoseDeltaModule):
        self._module = module

    @property
    def weight(self):
        return _LegacyPoseWeightView(self._module)


class _LegacyPoseWeightView:
    def __init__(self, module: _PoseDeltaModule):
        self._module = module

    @property
    def grad(self):
        trans_grad = self._module.trans.weight.grad
        rot_grad = self._module.rot.weight.grad
        if trans_grad is None or rot_grad is None:
            return None
        return torch.cat((trans_grad, rot_grad), dim=-1)


class CameraOptModule(_PoseDeltaModule):
    """Per-image camera pose optimization with SO(3) tangent corrections."""

    def __init__(
        self,
        n: int,
        reference_index: int | None = 0,
        rotation_mode: str = "so3",
    ):
        super().__init__(n, reference_index, rotation_mode)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Apply a camera-local pose correction to camera-to-world matrices."""
        assert camtoworlds.shape[:-2] == embed_ids.shape
        return torch.matmul(camtoworlds, self._delta_transform(embed_ids))


class CameraRigPoseModule(_PoseDeltaModule):
    """Shared rig-pose correction with fixed camera-to-rig extrinsics.

    The module derives a fixed rig model from the initial camera-to-world poses.
    Images sharing a frame id receive the same rig correction, while their
    physical-camera extrinsics remain tied together.
    """

    def __init__(
        self,
        camtoworlds: Tensor,
        frame_ids: Tensor,
        camera_ids: Tensor,
        reference_camera_id: int | None = None,
        reference_frame_id: int | None = None,
        rotation_mode: str = "so3",
    ):
        frame_values = torch.unique(frame_ids.detach().cpu(), sorted=True)
        camera_values = torch.unique(camera_ids.detach().cpu(), sorted=True)
        if frame_values.numel() == 0 or camera_values.numel() == 0:
            raise ValueError("A rig requires non-empty frame and camera ids")
        reference_camera_id = (
            int(camera_values[0]) if reference_camera_id is None else reference_camera_id
        )
        if reference_camera_id not in set(camera_values.tolist()):
            raise ValueError("reference_camera_id is not present in the dataset")
        if reference_frame_id is None:
            reference_frame_id = int(frame_values[0])
        if reference_frame_id not in set(frame_values.tolist()):
            raise ValueError("reference_frame_id is not present in the dataset")
        reference_frame_index = int(
            (frame_values == reference_frame_id).nonzero(as_tuple=False)[0]
        )
        super().__init__(len(frame_values), reference_frame_index, rotation_mode)

        base = camtoworlds.detach().cpu().float()
        frame_cpu = frame_ids.detach().cpu().long()
        camera_cpu = camera_ids.detach().cpu().long()
        base_rig = torch.eye(4).repeat(len(frame_values), 1, 1)
        camera_to_rig = torch.eye(4).repeat(len(camera_values), 1, 1)

        for frame_index, frame_value in enumerate(frame_values.tolist()):
            members = (frame_cpu == frame_value).nonzero(as_tuple=False).flatten()
            ref_members = members[camera_cpu[members] == reference_camera_id]
            selected = ref_members[0] if ref_members.numel() else members[0]
            base_rig[frame_index] = base[selected]

        ref_frame_members = (
            frame_cpu == int(frame_values[reference_frame_index])
        ).nonzero(as_tuple=False).flatten()
        for camera_index, camera_value in enumerate(camera_values.tolist()):
            members = ref_frame_members[camera_cpu[ref_frame_members] == camera_value]
            if members.numel():
                selected = members[0]
                camera_to_rig[camera_index] = (
                    torch.linalg.inv(base_rig[reference_frame_index]) @ base[selected]
                )
                continue
            found = False
            for frame_index, frame_value in enumerate(frame_values.tolist()):
                members = (frame_cpu == frame_value).nonzero(as_tuple=False).flatten()
                ref_members = members[camera_cpu[members] == reference_camera_id]
                cam_members = members[camera_cpu[members] == camera_value]
                if ref_members.numel() and cam_members.numel():
                    camera_to_rig[camera_index] = (
                        torch.linalg.inv(base[ref_members[0]]) @ base[cam_members[0]]
                    )
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Cannot derive camera-to-rig extrinsic for camera {camera_value}; "
                    "each rig camera must share a frame with the reference camera"
                )

        # Frames without the reference camera were initially represented by one
        # of their other cameras. Convert those representatives back to the rig
        # origin after all fixed camera-to-rig extrinsics are known.
        for frame_index, frame_value in enumerate(frame_values.tolist()):
            members = (frame_cpu == frame_value).nonzero(as_tuple=False).flatten()
            ref_members = members[camera_cpu[members] == reference_camera_id]
            if ref_members.numel():
                continue
            selected = members[0]
            camera_index = int(
                (camera_values == camera_cpu[selected]).nonzero(as_tuple=False)[0]
            )
            base_rig[frame_index] = (
                base[selected] @ torch.linalg.inv(camera_to_rig[camera_index])
            )

        self.register_buffer("frame_values", frame_values)
        self.register_buffer("camera_values", camera_values)
        self.register_buffer("base_rig_camtoworlds", base_rig)
        self.register_buffer("camera_to_rig", camera_to_rig)

    def _lookup(self, values: Tensor, query: Tensor) -> Tensor:
        query = query.to(device=values.device, dtype=values.dtype)
        indices = torch.searchsorted(values, query)
        if torch.any(indices >= values.numel()) or not torch.equal(
            values[indices.clamp_max(values.numel() - 1)], query
        ):
            raise ValueError("Rig received an unknown frame or camera id")
        return indices

    def forward(
        self,
        camtoworlds: Tensor,
        frame_ids: Tensor,
        camera_ids: Tensor,
    ) -> Tensor:
        assert camtoworlds.shape[:-2] == frame_ids.shape == camera_ids.shape
        frame_indices = self._lookup(self.frame_values, frame_ids)
        camera_indices = self._lookup(self.camera_values, camera_ids)
        rig_delta = self._delta_transform(frame_indices)
        return (
            self.base_rig_camtoworlds[frame_indices]
            @ rig_delta
            @ self.camera_to_rig[camera_indices]
        )


@dataclass(frozen=True)
class CameraRefinementStage:
    name: str
    pose: bool
    focal: bool
    principal: bool
    radial_low: bool
    radial_high: bool
    frozen: bool = False


class CameraRefinementSchedule:
    """Central stage policy for camera refinement."""

    def __init__(
        self,
        pose_start: int,
        focal_start: int,
        principal_start: int,
        radial_start: int,
        high_order_start: int,
        freeze_step: int,
    ):
        starts = [pose_start, focal_start, principal_start, radial_start]
        if any(step < 0 for step in starts):
            raise ValueError("Camera refinement start steps must be non-negative")
        if high_order_start < -1:
            raise ValueError("high_order_start must be -1 or non-negative")
        if freeze_step < -1:
            raise ValueError("freeze_step must be -1 or non-negative")
        self.pose_start = pose_start
        self.focal_start = focal_start
        self.principal_start = principal_start
        self.radial_start = radial_start
        self.high_order_start = high_order_start
        self.freeze_step = freeze_step

    def at(self, step: int, pose_enabled: bool, calibration_enabled: bool):
        if self.freeze_step >= 0 and step >= self.freeze_step:
            return CameraRefinementStage("frozen", False, False, False, False, False, True)
        pose = pose_enabled and step >= self.pose_start
        focal = calibration_enabled and step >= self.focal_start
        principal = calibration_enabled and step >= self.principal_start
        radial_low = calibration_enabled and step >= self.radial_start
        radial_high = (
            calibration_enabled
            and self.high_order_start >= 0
            and step >= self.high_order_start
        )
        if radial_high:
            name = "radial_high"
        elif radial_low:
            name = "radial_low"
        elif principal:
            name = "principal"
        elif focal:
            name = "focal"
        elif pose:
            name = "pose"
        else:
            name = "warmup"
        return CameraRefinementStage(name, pose, focal, principal, radial_low, radial_high)


class CameraCalibrationOptModule(torch.nn.Module):
    """Shared physical-camera calibration refinement.

    The four radial coefficients remain in one legacy-compatible table, but
    `radial_low`/`radial_high` expose low/high-order views and gradient controls
    keep k3/k4 frozen until the schedule explicitly enables them.
    """

    def __init__(self, n: int, shared_focal: bool = True, allow_aspect_ratio: bool = False):
        super().__init__()
        self.shared_focal = shared_focal
        self.allow_aspect_ratio = allow_aspect_ratio
        self.focal_log_scales = torch.nn.Embedding(n, 2)
        self.principal_offsets = torch.nn.Embedding(n, 2)
        self.radial_deltas = torch.nn.Embedding(n, 4)
        self.zero_init()

    @property
    def radial_low(self) -> Tensor:
        return self.radial_deltas.weight[..., :2]

    @property
    def radial_high(self) -> Tensor:
        return self.radial_deltas.weight[..., 2:]

    def zero_init(self):
        torch.nn.init.zeros_(self.focal_log_scales.weight)
        torch.nn.init.zeros_(self.principal_offsets.weight)
        torch.nn.init.zeros_(self.radial_deltas.weight)

    def forward(
        self, Ks: Tensor, radial_coeffs: Tensor, camera_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply calibration deltas shared by physical camera.

        Focal deltas are log-scales, so optimized focal lengths stay positive.
        Principal-point offsets are represented relative to the original focal
        lengths, which keeps their parameter scale independent of resolution.
        """
        assert Ks.shape[:-2] == camera_ids.shape
        assert radial_coeffs.shape[:-1] == camera_ids.shape
        assert radial_coeffs.shape[-1] == 4

        focal_log_scales = self.focal_log_scales(camera_ids)
        if self.shared_focal and not self.allow_aspect_ratio:
            focal_log_scales = focal_log_scales.mean(dim=-1, keepdim=True).expand_as(
                focal_log_scales
            )
        principal_offsets = self.principal_offsets(camera_ids)
        focal_lengths = torch.stack((Ks[..., 0, 0], Ks[..., 1, 1]), dim=-1)
        optimized_focals = focal_lengths * torch.exp(focal_log_scales)
        optimized_principal = torch.stack(
            (Ks[..., 0, 2], Ks[..., 1, 2]), dim=-1
        ) + principal_offsets * focal_lengths

        optimized_Ks = Ks.clone()
        optimized_Ks[..., 0, 0] = optimized_focals[..., 0]
        optimized_Ks[..., 1, 1] = optimized_focals[..., 1]
        optimized_Ks[..., 0, 2] = optimized_principal[..., 0]
        optimized_Ks[..., 1, 2] = optimized_principal[..., 1]
        optimized_radial = radial_coeffs + self.radial_deltas(camera_ids)
        return optimized_Ks, optimized_radial

    def apply_gradient_controls(
        self,
        focal_active: bool,
        principal_active: bool,
        radial_low_active: bool,
        radial_high_active: bool,
        radial_high_lr_scale: float = 1.0,
        aspect_lr_scale: float = 1.0,
    ):
        """Project stage-specific gradients before the calibration optimizer step."""
        focal_grad = self.focal_log_scales.weight.grad
        if focal_grad is not None and not focal_active:
            focal_grad.zero_()
        elif focal_grad is not None and self.shared_focal and not self.allow_aspect_ratio:
            focal_grad.copy_(focal_grad.mean(dim=-1, keepdim=True).expand_as(focal_grad))
        elif focal_grad is not None and self.allow_aspect_ratio:
            common = focal_grad.mean(dim=-1, keepdim=True)
            focal_grad.copy_(common + (focal_grad - common) * aspect_lr_scale)

        principal_grad = self.principal_offsets.weight.grad
        if principal_grad is not None and not principal_active:
            principal_grad.zero_()

        radial_grad = self.radial_deltas.weight.grad
        if radial_grad is not None:
            radial_grad[..., :2].mul_(1.0 if radial_low_active else 0.0)
            radial_grad[..., 2:].mul_(radial_high_lr_scale if radial_high_active else 0.0)

    @torch.no_grad()
    def project_parameters(self):
        if self.shared_focal and not self.allow_aspect_ratio:
            mean = self.focal_log_scales.weight.mean(dim=-1, keepdim=True)
            self.focal_log_scales.weight.copy_(mean.expand_as(self.focal_log_scales.weight))

    def prior_loss(
        self,
        focal_sigma: float,
        principal_sigma: float,
        radial_low_sigma: float,
        radial_high_sigma: float,
        aspect_sigma: float,
    ) -> Tensor:
        focal = self.focal_log_scales.weight
        common = focal.mean(dim=-1)
        loss = common.square().mean() / max(focal_sigma, 1e-8) ** 2
        if self.allow_aspect_ratio:
            aspect = (focal[..., 0] - focal[..., 1]) * 0.5
            loss = loss + aspect.square().mean() / max(aspect_sigma, 1e-8) ** 2
        principal = self.principal_offsets.weight
        loss = loss + principal.square().mean() / max(principal_sigma, 1e-8) ** 2
        low = self.radial_low
        high = self.radial_high
        loss = loss + low.square().mean() / max(radial_low_sigma, 1e-8) ** 2
        loss = loss + high.square().mean() / max(radial_high_sigma, 1e-8) ** 2
        return loss

    def monotonicity_loss(
        self,
        radial_coeffs: Tensor,
        theta_max: float,
        samples: int = 32,
        eps: float = 1e-3,
    ) -> Tensor:
        """Penalize folding of the OpenCV fisheye theta mapping."""
        theta = torch.linspace(
            0.0, theta_max, samples, device=radial_coeffs.device, dtype=radial_coeffs.dtype
        )
        theta2 = theta.square()
        k1, k2, k3, k4 = radial_coeffs.unbind(dim=-1)
        derivative = (
            1.0
            + 3.0 * k1[..., None] * theta2
            + 5.0 * k2[..., None] * theta2.square()
            + 7.0 * k3[..., None] * theta2.square() * theta2
            + 9.0 * k4[..., None] * theta2.square() * theta2.square()
        )
        return F.relu(eps - derivative).square().mean()

    @torch.no_grad()
    def metrics(self, base_focal: Tensor) -> dict[str, Tensor]:
        focal = self.focal_log_scales.weight
        if self.shared_focal and not self.allow_aspect_ratio:
            focal = focal.mean(dim=-1, keepdim=True).expand_as(focal)
        focal_change = torch.exp(focal) - 1.0
        principal_px = self.principal_offsets.weight * base_focal

        def stats(value: Tensor, suffix: str):
            flat = value.detach().reshape(-1).abs()
            return {
                f"{suffix}_mean": flat.mean(),
                f"{suffix}_p95": torch.quantile(flat, 0.95),
                f"{suffix}_max": flat.max(),
            }

        result = {}
        result.update(stats(focal_change * 100.0, "focal_change_pct"))
        result.update(stats(principal_px, "principal_change_px"))
        result.update(stats(self.radial_deltas.weight, "radial_delta"))
        return result


class PhotometricOptModule(torch.nn.Module):
    """Low-cost per-image RGB gain correction with a canonical zero-mean gauge."""

    def __init__(self, n: int):
        super().__init__()
        if n <= 0:
            raise ValueError("Photometric optimization requires at least one image")
        self.log_rgb_gains = torch.nn.Embedding(n, 3)
        torch.nn.init.zeros_(self.log_rgb_gains.weight)

    def centered_log_rgb_gains(self) -> Tensor:
        """Return log2 RGB gains whose per-channel dataset mean is exactly zero."""
        gains = self.log_rgb_gains.weight
        return gains - gains.mean(dim=0, keepdim=True)

    def forward(self, colors: Tensor, image_ids: Tensor | None) -> Tensor:
        """Apply per-image RGB gains, or identity for canonical/novel-view renders."""
        if image_ids is None:
            return colors
        image_ids = image_ids.long().reshape(-1)
        if colors.shape[0] != image_ids.numel():
            raise ValueError("colors batch dimension must match image_ids")
        log_gains = self.centered_log_rgb_gains()[image_ids]
        while log_gains.ndim < colors.ndim:
            log_gains = log_gains.unsqueeze(-2)
        return colors * torch.exp2(log_gains)

    def prior_loss(self) -> Tensor:
        """Keep corrections small while preserving the exact zero-mean gauge."""
        return self.centered_log_rgb_gains().square().mean()

    @torch.no_grad()
    def project_parameters(self) -> None:
        """Remove optimizer drift along the unobservable global-gain direction."""
        self.log_rgb_gains.weight.sub_(
            self.log_rgb_gains.weight.mean(dim=0, keepdim=True)
        )

    @torch.no_grad()
    def metrics(self) -> dict[str, Tensor]:
        log_gains = self.centered_log_rgb_gains()
        exposure = log_gains.mean(dim=-1)
        white_balance = log_gains - exposure[:, None]
        return {
            "exposure_abs_mean_log2": exposure.abs().mean(),
            "exposure_abs_max_log2": exposure.abs().max(),
            "white_balance_abs_mean_log2": white_balance.abs().mean(),
            "white_balance_abs_max_log2": white_balance.abs().max(),
        }


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img
