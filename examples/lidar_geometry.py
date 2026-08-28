from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gaussian_models import SurfacePriorData
from gsplat.utils import normalized_quat_to_rotmat


class LidarSurfelField:
    """A stateless LiDAR surface prior for freely moving 3D Gaussians.

    Reliable KNN-PCA samples are stored in a sparse voxel index. Each loss
    evaluation associates sampled Gaussians with nearby surfels from their
    current positions, so densification does not require inherited anchor state.
    """

    _MAX_LOSS_GAUSSIANS = 100_000

    def __init__(self, priors: SurfacePriorData, *, device: torch.device) -> None:
        points = priors.points_cpu
        normals = np.ascontiguousarray(
            priors.normals.detach().float().cpu().numpy()
        )
        confidence = np.ascontiguousarray(
            priors.confidence.detach().float().cpu().numpy()
        )
        radius = np.ascontiguousarray(
            priors.radius.detach().float().cpu().numpy()
        )
        scales = np.ascontiguousarray(
            priors.scales.detach().float().cpu().numpy()
        )
        local_scale = np.sqrt(
            np.maximum(scales[:, 0] * scales[:, 1], np.finfo(np.float32).eps)
        )

        reliable = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(normals).all(axis=1)
            & np.isfinite(confidence)
            & np.isfinite(radius)
            & np.isfinite(local_scale)
            & (confidence > 0.0)
            & (radius > 0.0)
            & (local_scale > 0.0)
        )
        reliable_ids = np.flatnonzero(reliable)
        if reliable_ids.size == 0:
            raise ValueError("KNN-PCA produced no reliable LiDAR surfels")

        # The KNN support radius is a data-derived spatial index resolution. It
        # keeps query support tied to the input sampling density without adding
        # another user-facing threshold.
        voxel_size = float(np.median(radius[reliable_ids]))
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("LiDAR surfel voxel size must be finite and positive")

        grid_coords = np.floor(points[reliable_ids] / voxel_size).astype(np.int64)
        grid_min = grid_coords.min(axis=0)
        relative_coords = grid_coords - grid_min
        grid_dims = relative_coords.max(axis=0) + 1
        if int(np.prod(grid_dims.astype(object))) > np.iinfo(np.int64).max:
            raise ValueError("LiDAR surfel grid is too large for int64 keys")

        linear_keys = (
            (relative_coords[:, 0] * grid_dims[1] + relative_coords[:, 1])
            * grid_dims[2]
            + relative_coords[:, 2]
        )
        # lexsort uses the last key as primary, so keys are ascending and
        # confidence is descending within each voxel.
        order = np.lexsort((-confidence[reliable_ids], linear_keys))
        sorted_keys = linear_keys[order]
        keep = np.empty(sorted_keys.shape, dtype=bool)
        keep[0] = True
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
        surfel_ids = reliable_ids[order[keep]]

        tensor_kwargs = {"device": device}
        self._keys = torch.from_numpy(sorted_keys[keep]).to(
            dtype=torch.int64, **tensor_kwargs
        )
        self._points = torch.from_numpy(points[surfel_ids]).to(
            dtype=torch.float32, **tensor_kwargs
        )
        self._normals = F.normalize(
            torch.from_numpy(normals[surfel_ids]).to(
                dtype=torch.float32, **tensor_kwargs
            ),
            dim=-1,
            eps=1e-12,
        )
        self._confidence = torch.from_numpy(confidence[surfel_ids]).to(
            dtype=torch.float32, **tensor_kwargs
        )
        self._local_scale = torch.from_numpy(local_scale[surfel_ids]).to(
            dtype=torch.float32, **tensor_kwargs
        )
        self._grid_min = torch.from_numpy(grid_min).to(
            dtype=torch.int64, **tensor_kwargs
        )
        self._grid_dims = torch.from_numpy(grid_dims).to(
            dtype=torch.int64, **tensor_kwargs
        )
        self._neighbor_offsets = torch.tensor(
            [
                (x, y, z)
                for x in (-1, 0, 1)
                for y in (-1, 0, 1)
                for z in (-1, 0, 1)
            ],
            dtype=torch.int64,
            **tensor_kwargs,
        )
        self.voxel_size = voxel_size
        self.num_surfels = len(surfel_ids)
        self.num_reliable_points = len(reliable_ids)

    def _sample_visible_ids(self, visible: Tensor, *, step: int) -> Tensor:
        ids = torch.where(visible)[0]
        if ids.numel() <= self._MAX_LOSS_GAUSSIANS:
            return ids
        stride = (ids.numel() + self._MAX_LOSS_GAUSSIANS - 1) // (
            self._MAX_LOSS_GAUSSIANS
        )
        offset = step % stride
        return ids[offset::stride][: self._MAX_LOSS_GAUSSIANS]

    def _query_nearby_surfels(
        self, means: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        query_coords = torch.floor(means.detach() / self.voxel_size).to(torch.int64)
        query_coords = query_coords - self._grid_min
        candidate_coords = query_coords[:, None, :] + self._neighbor_offsets[None]
        in_grid = (
            (candidate_coords >= 0) & (candidate_coords < self._grid_dims)
        ).all(dim=-1)
        candidate_keys = (
            (
                candidate_coords[..., 0] * self._grid_dims[1]
                + candidate_coords[..., 1]
            )
            * self._grid_dims[2]
            + candidate_coords[..., 2]
        ).contiguous()

        slots = torch.searchsorted(self._keys, candidate_keys)
        safe_slots = slots.clamp_max(len(self._keys) - 1)
        matched = in_grid & (slots < len(self._keys)) & (
            self._keys[safe_slots] == candidate_keys
        )

        candidate_points = self._points[safe_slots]
        distances_sq = (candidate_points - means.detach()[:, None, :]).square().sum(
            dim=-1
        )
        distances_sq = distances_sq.masked_fill(~matched, torch.inf)
        nearest_distance_sq, nearest_local_id = distances_sq.min(dim=1)
        valid = torch.isfinite(nearest_distance_sq)
        nearest_slot = safe_slots.gather(1, nearest_local_id[:, None]).squeeze(1)
        return (
            self._points[nearest_slot],
            self._normals[nearest_slot],
            self._confidence[nearest_slot],
            self._local_scale[nearest_slot],
            valid,
        )

    def compute_loss(
        self,
        splats: Dict[str, Tensor],
        visible: Tensor,
        *,
        step: int,
        compute_stats: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return the expected Gaussian distance to nearby LiDAR planes."""
        means_all = splats["means"]
        if visible.shape != means_all.shape[:1]:
            raise ValueError("Visible mask must match the Gaussian count")

        sample_ids = self._sample_visible_ids(visible, step=step)
        zero = means_all.sum() * 0.0
        if sample_ids.numel() == 0:
            return zero, {
                "valid_ratio": zero.detach(),
                "sample_ratio": zero.detach(),
            }

        means = means_all[sample_ids]
        points, normals, confidence, local_scale, valid = (
            self._query_nearby_surfels(means)
        )
        sample_ratio = means.new_tensor(
            sample_ids.numel() / max(len(means_all), 1)
        )
        if not valid.any():
            return zero, {
                "valid_ratio": valid.float().mean(),
                "sample_ratio": sample_ratio,
            }

        valid_ratio = valid.float().mean()
        sample_ids = sample_ids[valid]
        means = means[valid]
        points = points[valid]
        normals = normals[valid]
        confidence = confidence[valid]
        local_scale = local_scale[valid]
        rotations = normalized_quat_to_rotmat(
            F.normalize(splats["quats"][sample_ids], dim=-1, eps=1e-12)
        )
        scales = torch.exp(splats["scales"][sample_ids])
        plane_distance = ((means - points) * normals).sum(dim=-1)
        normal_in_gaussian_axes = torch.einsum(
            "nij,ni->nj", rotations, normals
        )
        normal_variance = (
            normal_in_gaussian_axes.square() * scales.square()
        ).sum(dim=-1)
        expected_distance = torch.sqrt(
            plane_distance.square() + normal_variance + 1e-12
        )
        normalized_distance = expected_distance / local_scale.clamp_min(1e-6)
        per_gaussian = F.smooth_l1_loss(
            normalized_distance,
            torch.zeros_like(normalized_distance),
            reduction="none",
        )
        loss = (per_gaussian * confidence).sum() / confidence.sum().clamp_min(1e-6)

        stats: Dict[str, Tensor] = {
            "valid_ratio": valid_ratio.detach(),
            "sample_ratio": sample_ratio,
        }
        if compute_stats:
            valid_expected = expected_distance.detach()
            valid_center = plane_distance.detach().abs()
            valid_thickness = normal_variance.detach().sqrt()
            stats.update(
                {
                    "distance_p50": torch.quantile(valid_expected, 0.50),
                    "distance_p90": torch.quantile(valid_expected, 0.90),
                    "center_p90": torch.quantile(valid_center, 0.90),
                    "thickness_p90": torch.quantile(valid_thickness, 0.90),
                }
            )
        return loss, stats
