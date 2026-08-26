import math
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gaussian_models import SurfacePriorData
from gsplat.utils import normalized_quat_to_rotmat


def _multiply_wxyz_quaternions(left: Tensor, right: Tensor) -> Tensor:
    """Compose wxyz quaternions so the left rotation is applied last."""
    left_w, left_xyz = left[..., :1], left[..., 1:]
    right_w, right_xyz = right[..., :1], right[..., 1:]
    return torch.cat(
        [
            left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True),
            left_w * right_xyz
            + right_w * left_xyz
            + torch.cross(left_xyz, right_xyz, dim=-1),
        ],
        dim=-1,
    )


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
        offset_bound: float,
        offset_bound_min_confidence: float,
        tilt_bound_deg: float,
        tilt_min_confidence: float,
        tilt_min_anisotropy: float,
        thickness_bound_ratio: float,
        thickness_bound_min_confidence: float,
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
        if not 0.0 < offset_bound < max_distance:
            raise ValueError(
                "Surface offset bound must be positive and below max distance"
            )
        if not 0.0 <= offset_bound_min_confidence <= 1.0:
            raise ValueError("Surface offset confidence must be in [0, 1]")
        if not 0.0 < tilt_bound_deg < 90.0:
            raise ValueError("Surface tilt bound must be in (0, 90) degrees")
        if not 0.0 <= tilt_min_confidence <= 1.0:
            raise ValueError("Surface tilt confidence must be in [0, 1]")
        if not 0.0 <= tilt_min_anisotropy <= 1.0:
            raise ValueError("Surface tilt anisotropy must be in [0, 1]")
        if not 0.0 < thickness_bound_ratio < 1.0:
            raise ValueError("Surface thickness bound ratio must be in (0, 1)")
        if not 0.0 <= thickness_bound_min_confidence <= 1.0:
            raise ValueError("Surface thickness bound confidence must be in [0, 1]")
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
        self.offset_bound = float(offset_bound)
        self.offset_bound_min_confidence = float(offset_bound_min_confidence)
        self.tilt_bound_deg = float(tilt_bound_deg)
        self.tilt_min_confidence = float(tilt_min_confidence)
        self.tilt_min_anisotropy = float(tilt_min_anisotropy)
        self.thickness_bound_ratio = float(thickness_bound_ratio)
        self.thickness_bound_min_confidence = float(
            thickness_bound_min_confidence
        )
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
    def project_bounded_offset(
        self,
        splats: Dict[str, Tensor],
        state: Dict[str, Any],
        *,
        compute_stats: bool = False,
    ) -> Dict[str, Tensor]:
        """Clamp reliable Gaussian centers along their anchor plane normal."""
        means = splats["means"]
        if len(state[self._XYZ_KEY]) != len(means):
            raise RuntimeError(
                "Surface anchor state is out of sync with Gaussian parameters"
            )

        anchor_xyz = state[self._XYZ_KEY]
        anchor_normal = F.normalize(
            state[self._NORMAL_KEY], dim=-1, eps=1e-12
        )
        offset = means.detach() - anchor_xyz
        plane_offset = (offset * anchor_normal).sum(dim=-1)
        anchor_distance = torch.linalg.vector_norm(offset, dim=-1)
        eligible = (
            state[self._VALID_KEY]
            & (state[self._CONF_KEY] >= self.offset_bound_min_confidence)
            & (anchor_distance <= self.max_distance)
        )
        outside = eligible & (plane_offset.abs() > self.offset_bound)
        outside_ids = torch.where(outside)[0]
        eligible_count = eligible.sum()
        outside_count = outside.sum()
        total = max(len(means), 1)
        zero = means.new_zeros(())
        stats = {
            "offset_bound_eligible_count": eligible_count,
            "offset_bound_eligible_ratio": eligible_count.to(means.dtype) / total,
            "offset_bound_projected_count": outside_count,
            "offset_bound_projected_ratio": outside_count.to(means.dtype) / total,
            "offset_bound_projected_of_eligible": (
                outside_count.to(means.dtype)
                / eligible_count.to(means.dtype).clamp_min(1.0)
            ),
            "offset_bound_pre_p90": zero,
            "offset_bound_post_max": zero,
        }
        if outside_ids.numel() == 0:
            return stats

        projected_offset = plane_offset[outside_ids].clamp(
            -self.offset_bound, self.offset_bound
        )
        correction = plane_offset[outside_ids] - projected_offset
        splats["means"][outside_ids] -= correction[:, None] * anchor_normal[
            outside_ids
        ]

        if compute_stats:
            pre_offset = plane_offset[outside_ids].abs()
            if pre_offset.numel() > 100_000:
                stride = (pre_offset.numel() + 99_999) // 100_000
                pre_offset = pre_offset[::stride][:100_000]
            stats["offset_bound_pre_p90"] = torch.quantile(pre_offset, 0.90)
            post_offset = (
                (splats["means"].detach()[outside_ids] - anchor_xyz[outside_ids])
                * anchor_normal[outside_ids]
            ).sum(dim=-1).abs()
            stats["offset_bound_post_max"] = post_offset.max()
        return stats

    @torch.no_grad()
    def project_bounded_thickness(
        self,
        splats: Dict[str, Tensor],
        state: Dict[str, Any],
        *,
        compute_stats: bool = False,
    ) -> Dict[str, Tensor]:
        """Give reliable surface Gaussians a unique, bounded shortest axis."""
        means = splats["means"]
        if len(state[self._XYZ_KEY]) != len(means):
            raise RuntimeError(
                "Surface anchor state is out of sync with Gaussian parameters"
            )

        log_scales = splats["scales"].detach()
        sorted_log_scales, sorted_axes = torch.sort(log_scales, dim=-1)
        thickness = torch.exp(
            sorted_log_scales[:, 0] - sorted_log_scales[:, 1]
        )
        anchor_distance = torch.linalg.vector_norm(
            means.detach() - state[self._XYZ_KEY], dim=-1
        )
        eligible = (
            state[self._VALID_KEY]
            & (state[self._CONF_KEY] >= self.thickness_bound_min_confidence)
            & (anchor_distance <= self.max_distance)
        )
        outside = eligible & (thickness > self.thickness_bound_ratio)
        outside_ids = torch.where(outside)[0]
        eligible_count = eligible.sum()
        outside_count = outside.sum()
        total = max(len(means), 1)
        zero = means.new_zeros(())
        stats = {
            "thickness_bound_eligible_count": eligible_count,
            "thickness_bound_eligible_ratio": eligible_count.to(means.dtype)
            / total,
            "thickness_bound_clamped_count": outside_count,
            "thickness_bound_clamped_ratio": outside_count.to(means.dtype)
            / total,
            "thickness_bound_clamped_of_eligible": (
                outside_count.to(means.dtype)
                / eligible_count.to(means.dtype).clamp_min(1.0)
            ),
            "thickness_bound_pre_p90": zero,
            "thickness_bound_post_max": zero,
        }
        if outside_ids.numel() == 0:
            return stats

        min_axes = sorted_axes[outside_ids, 0]
        target_log_scales = (
            sorted_log_scales[outside_ids, 1]
            + math.log(self.thickness_bound_ratio)
        )
        splats["scales"][outside_ids, min_axes] = target_log_scales

        if compute_stats:
            pre_thickness = thickness[outside_ids]
            if pre_thickness.numel() > 100_000:
                stride = (pre_thickness.numel() + 99_999) // 100_000
                pre_thickness = pre_thickness[::stride][:100_000]
            stats["thickness_bound_pre_p90"] = torch.quantile(
                pre_thickness, 0.90
            )
            projected_log_scales = torch.sort(
                splats["scales"].detach()[outside_ids], dim=-1
            ).values
            stats["thickness_bound_post_max"] = torch.exp(
                projected_log_scales[:, 0] - projected_log_scales[:, 1]
            ).max()
        return stats

    @torch.no_grad()
    def project_bounded_tilt(
        self,
        splats: Dict[str, Tensor],
        state: Dict[str, Any],
        *,
        compute_stats: bool = False,
    ) -> Dict[str, Tensor]:
        """Project reliable, surface-like Gaussian normals into the anchor cone.

        The shortest scale axis defines the unoriented Gaussian normal. A minimal
        world-space rotation moves only normals outside the configured cone onto
        its boundary, preserving the Gaussian's in-plane orientation as much as
        possible.
        """
        means = splats["means"]
        if len(state[self._NORMAL_KEY]) != len(means):
            raise RuntimeError(
                "Surface anchor state is out of sync with Gaussian parameters"
            )

        scales = torch.exp(splats["scales"].detach())
        sorted_scales, sorted_axes = torch.sort(scales, dim=-1)
        thickness = sorted_scales[:, 0] / sorted_scales[:, 1].clamp_min(1e-8)
        anisotropy = 1.0 - thickness
        anchor_distance = torch.linalg.vector_norm(
            means.detach() - state[self._XYZ_KEY], dim=-1
        )
        eligible = (
            state[self._VALID_KEY]
            & (state[self._CONF_KEY] >= self.tilt_min_confidence)
            & (anchor_distance <= self.max_distance)
            & (anisotropy >= self.tilt_min_anisotropy)
        )
        eligible_ids = torch.where(eligible)[0]
        total = max(len(means), 1)
        eligible_count = eligible.sum()
        zero = means.new_zeros(())
        stats = {
            "tilt_eligible_count": eligible_count,
            "tilt_eligible_ratio": eligible_count.to(means.dtype) / total,
            "tilt_projected_count": eligible_count.new_zeros(()),
            "tilt_projected_ratio": zero,
            "tilt_projected_of_eligible": zero,
            "tilt_pre_p90": zero,
            "tilt_post_max": zero,
        }
        if eligible_ids.numel() == 0:
            return stats

        quats = F.normalize(
            splats["quats"].detach()[eligible_ids], dim=-1, eps=1e-12
        )
        rotations = normalized_quat_to_rotmat(quats)
        min_axes = sorted_axes[eligible_ids, 0]
        gaussian_normal = torch.gather(
            rotations,
            dim=2,
            index=min_axes[:, None, None].expand(-1, 3, 1),
        ).squeeze(-1)
        anchor_normal = F.normalize(
            state[self._NORMAL_KEY][eligible_ids], dim=-1, eps=1e-12
        )
        signed_cosine = (gaussian_normal * anchor_normal).sum(dim=-1)
        target_normal = torch.where(
            (signed_cosine >= 0.0)[:, None], anchor_normal, -anchor_normal
        )
        normal_cosine = signed_cosine.abs().clamp(0.0, 1.0)
        angle = torch.acos(normal_cosine)
        bound_radians = math.radians(self.tilt_bound_deg)
        outside = angle > bound_radians
        outside_local_ids = torch.where(outside)[0]
        outside_count = outside.sum()
        stats["tilt_projected_count"] = outside_count
        stats["tilt_projected_ratio"] = outside_count.to(means.dtype) / total
        stats["tilt_projected_of_eligible"] = outside_count.to(means.dtype) / (
            eligible_count.to(means.dtype).clamp_min(1.0)
        )
        if outside_local_ids.numel() == 0:
            return stats

        source = gaussian_normal[outside_local_ids]
        target = target_normal[outside_local_ids]
        rotation_axis = F.normalize(
            torch.cross(source, target, dim=-1), dim=-1, eps=1e-12
        )
        correction_angle = angle[outside_local_ids] - bound_radians
        half_angle = correction_angle * 0.5
        correction_quat = torch.cat(
            [
                torch.cos(half_angle)[:, None],
                rotation_axis * torch.sin(half_angle)[:, None],
            ],
            dim=-1,
        )
        projected_quats = F.normalize(
            _multiply_wxyz_quaternions(
                correction_quat, quats[outside_local_ids]
            ),
            dim=-1,
            eps=1e-12,
        )
        projected_ids = eligible_ids[outside_local_ids]
        splats["quats"][projected_ids] = projected_quats

        if compute_stats:
            pre_angles = torch.rad2deg(angle[outside_local_ids])
            if pre_angles.numel() > 100_000:
                stride = (pre_angles.numel() + 99_999) // 100_000
                pre_angles = pre_angles[::stride][:100_000]
            stats["tilt_pre_p90"] = torch.quantile(pre_angles, 0.90)

            projected_rotations = normalized_quat_to_rotmat(projected_quats)
            projected_normals = torch.gather(
                projected_rotations,
                dim=2,
                index=min_axes[outside_local_ids, None, None].expand(-1, 3, 1),
            ).squeeze(-1)
            post_cosine = (projected_normals * target).sum(dim=-1).clamp(
                -1.0, 1.0
            )
            stats["tilt_post_max"] = torch.rad2deg(
                torch.acos(post_cosine)
            ).max()
        return stats

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
