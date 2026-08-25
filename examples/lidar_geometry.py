from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gaussian_models import SurfacePriorData
from gsplat.utils import normalized_quat_to_rotmat


class LidarSurfaceGeometry:
    """Keep strategy-managed 3D Gaussians near a fixed PLY surface manifold."""

    _XYZ_KEY = "surface_anchor_xyz"
    _NORMAL_KEY = "surface_anchor_normal"
    _CONF_KEY = "surface_anchor_confidence"
    _RADIUS_KEY = "surface_anchor_radius"
    _VALID_KEY = "surface_anchor_valid"
    _OFF_COUNT_KEY = "surface_offsurface_count"
    _PRUNE_MASK_KEY = "surface_prune_mask"

    def __init__(
        self,
        priors: SurfacePriorData,
        *,
        distance_scale: float,
        dead_zone: float,
        max_distance: float,
        min_confidence: float,
        refresh_distance: float,
        max_loss_gaussians: int,
        normal_anisotropy_min: float,
        normal_anisotropy_max: float,
        thickness_ratio: float,
        prune_enabled: bool,
        prune_distance: float,
        prune_min_confidence: float,
        prune_min_opacity: float,
        prune_patience: int,
        query_chunk_size: int = 262_144,
    ) -> None:
        if min(distance_scale, max_distance, refresh_distance) <= 0.0:
            raise ValueError(
                "Surface distance scale, max distance, and refresh distance "
                "must be positive"
            )
        if dead_zone < 0.0:
            raise ValueError("Surface dead zone must be non-negative")
        if max_distance <= dead_zone:
            raise ValueError("Surface max distance must exceed the dead zone")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Surface minimum confidence must be in [0, 1]")
        if query_chunk_size < 1:
            raise ValueError("Surface query chunk size must be positive")
        if max_loss_gaussians < 1:
            raise ValueError("Surface loss sample count must be positive")
        if not 0.0 <= normal_anisotropy_min < normal_anisotropy_max <= 1.0:
            raise ValueError("Surface normal anisotropy bounds must satisfy 0 <= min < max <= 1")
        if not 0.0 < thickness_ratio < 1.0:
            raise ValueError("Surface thickness ratio must be in (0, 1)")
        if prune_distance <= dead_zone:
            raise ValueError("Surface prune distance must exceed the dead zone")
        if not 0.0 <= prune_min_confidence <= 1.0:
            raise ValueError("Surface prune confidence must be in [0, 1]")
        if not 0.0 <= prune_min_opacity <= 1.0:
            raise ValueError("Surface prune opacity must be in [0, 1]")
        if prune_patience < 1:
            raise ValueError("Surface prune patience must be positive")

        self.distance_scale = float(distance_scale)
        self.dead_zone = float(dead_zone)
        self.max_distance = float(max_distance)
        self.min_confidence = float(min_confidence)
        self.refresh_distance = float(refresh_distance)
        self.max_loss_gaussians = int(max_loss_gaussians)
        self.normal_anisotropy_min = float(normal_anisotropy_min)
        self.normal_anisotropy_max = float(normal_anisotropy_max)
        self.thickness_ratio = float(thickness_ratio)
        self.prune_enabled = bool(prune_enabled)
        self.prune_distance = float(prune_distance)
        self.prune_min_confidence = float(prune_min_confidence)
        self.prune_min_opacity = float(prune_min_opacity)
        self.prune_patience = int(prune_patience)
        self.query_chunk_size = int(query_chunk_size)

        self._points_cpu = priors.points_cpu
        self._normals_cpu = np.ascontiguousarray(
            priors.normals.detach().float().cpu().numpy()
        )
        self._confidence_cpu = np.ascontiguousarray(
            priors.confidence.detach().float().cpu().numpy()
        )
        self._radius_cpu = np.ascontiguousarray(
            priors.radius.detach().float().cpu().numpy()
        )
        self._neighbors_model = priors.neighbors_model

    @torch.no_grad()
    def attach_to_strategy_state(
        self,
        state: Dict[str, Any],
        means: Tensor,
        *,
        world_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Attach per-Gaussian anchors that gsplat will grow/prune with params."""
        reference_indices = np.arange(
            world_rank, len(self._points_cpu), world_size, dtype=np.int64
        )
        if len(reference_indices) != len(means):
            raise ValueError(
                "Initial surface anchors do not match the distributed Gaussian count: "
                f"{len(reference_indices)} anchors for {len(means)} Gaussians"
            )

        device, dtype = means.device, means.dtype
        normals = torch.from_numpy(self._normals_cpu[reference_indices]).to(
            device=device, dtype=dtype
        )
        confidence = torch.from_numpy(self._confidence_cpu[reference_indices]).to(
            device=device, dtype=dtype
        )
        radius = torch.from_numpy(self._radius_cpu[reference_indices]).to(
            device=device, dtype=dtype
        )
        state[self._XYZ_KEY] = means.detach().clone()
        state[self._NORMAL_KEY] = F.normalize(normals, dim=-1, eps=1e-12)
        state[self._CONF_KEY] = confidence
        state[self._RADIUS_KEY] = radius
        state[self._VALID_KEY] = confidence >= self.min_confidence
        state[self._OFF_COUNT_KEY] = torch.zeros(
            len(means), dtype=torch.int16, device=device
        )
        state[self._PRUNE_MASK_KEY] = torch.zeros(
            len(means), dtype=torch.bool, device=device
        )

    def _loss_sample_indices(self, count: int, step: int, device: torch.device):
        """Deterministically rotate through large Gaussian sets without randperm."""
        if count <= self.max_loss_gaussians:
            return None
        stride = (count + self.max_loss_gaussians - 1) // self.max_loss_gaussians
        offset = step % stride
        return torch.arange(offset, count, stride, device=device)[: self.max_loss_gaussians]

    def compute_loss(
        self,
        splats: Dict[str, Tensor],
        state: Dict[str, Any],
        *,
        step: int,
        compute_stats: bool = False,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        means = splats["means"]
        sample_ids = self._loss_sample_indices(len(means), step, means.device)

        def sampled(value: Tensor) -> Tensor:
            return value if sample_ids is None else value[sample_ids]

        anchor_xyz = sampled(state[self._XYZ_KEY])
        anchor_normal = sampled(state[self._NORMAL_KEY])
        anchor_confidence = sampled(state[self._CONF_KEY])
        if len(anchor_xyz) != len(means):
            if sample_ids is None:
                raise RuntimeError(
                    "Surface anchor state is out of sync with Gaussian parameters"
                )
            means = means[sample_ids]

        offset = means - anchor_xyz
        plane_distance = (offset * anchor_normal).sum(dim=-1)
        anchor_distance = torch.linalg.vector_norm(offset, dim=-1)
        valid = (
            sampled(state[self._VALID_KEY])
            & (anchor_confidence >= self.min_confidence)
            & (anchor_distance <= self.max_distance)
        )
        weight = anchor_confidence * valid.to(anchor_confidence.dtype)
        residual = (
            (plane_distance.abs() - self.dead_zone).clamp_min(0.0)
            / self.distance_scale
        )
        per_gaussian = F.smooth_l1_loss(
            residual, torch.zeros_like(residual), reduction="none"
        )
        weight_sum = weight.sum()
        surface_loss = (per_gaussian * weight).sum() / weight_sum.clamp_min(1e-6)

        scales = torch.exp(sampled(splats["scales"]))
        sorted_scales, sorted_axes = torch.sort(scales, dim=-1)
        thickness = sorted_scales[:, 0] / sorted_scales[:, 1].clamp_min(1e-8)
        anisotropy = 1.0 - thickness
        anisotropy_weight = (
            (anisotropy - self.normal_anisotropy_min)
            / (self.normal_anisotropy_max - self.normal_anisotropy_min)
        ).clamp(0.0, 1.0)

        rotations = normalized_quat_to_rotmat(
            F.normalize(sampled(splats["quats"]), dim=-1, eps=1e-12)
        )
        min_axes = sorted_axes[:, 0]
        gaussian_normal = torch.gather(
            rotations,
            dim=2,
            index=min_axes[:, None, None].expand(-1, 3, 1),
        ).squeeze(-1)
        normal_cosine = (gaussian_normal * anchor_normal).sum(dim=-1).abs().clamp_max(1.0)
        normal_weight = weight * anisotropy_weight
        normal_loss = ((1.0 - normal_cosine) * normal_weight).sum() / (
            normal_weight.sum().clamp_min(1e-6)
        )

        thickness_residual = (thickness - self.thickness_ratio).clamp_min(0.0)
        thickness_loss = (thickness_residual * weight).sum() / weight_sum.clamp_min(
            1e-6
        )

        losses = {
            "surface": surface_loss,
            "normal": normal_loss,
            "thickness": thickness_loss,
        }

        stats: Dict[str, Tensor] = {
            "valid_ratio": valid.float().mean().detach(),
            "loss_sample_ratio": means.new_tensor(len(means) / len(splats["means"])),
        }
        if compute_stats:
            deviations = plane_distance.detach().abs()[valid]
            if deviations.numel() > 0:
                # Quantiles sort their input; cap monitoring cost for multi-million
                # Gaussian scenes with a deterministic, evenly spaced sample.
                if deviations.numel() > 100_000:
                    stride = (deviations.numel() + 99_999) // 100_000
                    deviations = deviations[::stride][:100_000]
                valid_thickness = thickness.detach()[valid]
                valid_angles = torch.rad2deg(
                    torch.acos(normal_cosine.detach()[valid].clamp(0.0, 1.0))
                )
                if valid_thickness.numel() > 100_000:
                    stride = (valid_thickness.numel() + 99_999) // 100_000
                    valid_thickness = valid_thickness[::stride][:100_000]
                    valid_angles = valid_angles[::stride][:100_000]
                stats.update(
                    {
                        "rmse": deviations.square().mean().sqrt(),
                        "p50": torch.quantile(deviations, 0.50),
                        "p90": torch.quantile(deviations, 0.90),
                        "p95": torch.quantile(deviations, 0.95),
                        "off_surface_ratio": (
                            deviations > self.dead_zone
                        ).float().mean(),
                        "thickness_p50": torch.quantile(valid_thickness, 0.50),
                        "thickness_p90": torch.quantile(valid_thickness, 0.90),
                        "normal_angle_p50": torch.quantile(valid_angles, 0.50),
                        "normal_angle_p90": torch.quantile(valid_angles, 0.90),
                        "prune_candidate_ratio": state[
                            self._PRUNE_MASK_KEY
                        ].float().mean(),
                    }
                )
            else:
                zero = means.detach().new_zeros(())
                stats.update(
                    {
                        "rmse": zero,
                        "p50": zero,
                        "p90": zero,
                        "p95": zero,
                        "off_surface_ratio": zero,
                        "thickness_p50": zero,
                        "thickness_p90": zero,
                        "normal_angle_p50": zero,
                        "normal_angle_p90": zero,
                        "prune_candidate_ratio": state[
                            self._PRUNE_MASK_KEY
                        ].float().mean(),
                    }
                )
        return losses, stats

    @torch.no_grad()
    def update_prune_state(
        self,
        splats: Dict[str, Tensor],
        state: Dict[str, Any],
    ) -> int:
        """Mark only persistent, opaque deviations from reliable PLY planes."""
        if not self.prune_enabled:
            state[self._OFF_COUNT_KEY].zero_()
            state[self._PRUNE_MASK_KEY].zero_()
            return 0

        offset = splats["means"].detach() - state[self._XYZ_KEY]
        plane_distance = (offset * state[self._NORMAL_KEY]).sum(dim=-1).abs()
        opacity = torch.sigmoid(splats["opacities"].detach().flatten())
        off_surface = (
            (state[self._CONF_KEY] >= self.prune_min_confidence)
            & (opacity >= self.prune_min_opacity)
            & (plane_distance > self.prune_distance)
        )
        counts = state[self._OFF_COUNT_KEY]
        incremented = (counts + 1).clamp_max(self.prune_patience)
        counts.copy_(torch.where(off_surface, incremented, torch.zeros_like(counts)))
        state[self._PRUNE_MASK_KEY].copy_(counts >= self.prune_patience)
        return int(state[self._PRUNE_MASK_KEY].sum().item())

    @torch.no_grad()
    def refresh_anchors(
        self,
        means: Tensor,
        state: Dict[str, Any],
        *,
        force: bool = False,
    ) -> int:
        """Reassociate only Gaussians that have left their inherited support."""
        if self._neighbors_model is None or len(means) == 0:
            return 0

        anchor_xyz = state[self._XYZ_KEY]
        displacement = torch.linalg.vector_norm(means.detach() - anchor_xyz, dim=-1)
        support_threshold = torch.minimum(
            state[self._RADIUS_KEY] * 2.0,
            torch.full_like(displacement, self.refresh_distance),
        ).clamp_min(self.dead_zone)
        candidates = torch.ones_like(displacement, dtype=torch.bool) if force else (
            displacement > support_threshold
        )
        candidate_ids = torch.where(candidates)[0]
        if candidate_ids.numel() == 0:
            return 0

        device, dtype = means.device, means.dtype
        for start in range(0, candidate_ids.numel(), self.query_chunk_size):
            ids = candidate_ids[start : start + self.query_chunk_size]
            query = np.ascontiguousarray(means.detach()[ids].float().cpu().numpy())
            distances, indices = self._neighbors_model.kneighbors(
                query, n_neighbors=1, return_distance=True
            )
            nearest = indices[:, 0]
            nearest_distance = torch.from_numpy(distances[:, 0]).to(
                device=device, dtype=dtype
            )
            confidence = torch.from_numpy(self._confidence_cpu[nearest]).to(
                device=device, dtype=dtype
            )
            state[self._XYZ_KEY][ids] = torch.from_numpy(
                self._points_cpu[nearest]
            ).to(device=device, dtype=dtype)
            state[self._NORMAL_KEY][ids] = F.normalize(
                torch.from_numpy(self._normals_cpu[nearest]).to(
                    device=device, dtype=dtype
                ),
                dim=-1,
                eps=1e-12,
            )
            state[self._CONF_KEY][ids] = confidence
            state[self._RADIUS_KEY][ids] = torch.from_numpy(
                self._radius_cpu[nearest]
            ).to(device=device, dtype=dtype)
            state[self._VALID_KEY][ids] = (
                (nearest_distance <= self.max_distance)
                & (confidence >= self.min_confidence)
            )
        return int(candidate_ids.numel())
