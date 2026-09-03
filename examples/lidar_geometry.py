import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gaussian_models import SurfacePriorData
from gsplat.utils import normalized_quat_to_rotmat


class LidarSurfelField:
    """Stateless, density-tolerant LiDAR surface supervision for 3D Gaussians."""

    update_every = 4
    warmup_steps = 500

    _MAX_LOSS_GAUSSIANS = 32_768
    _NUM_PLANES = 4
    _QUERY_CHUNK_SIZE = 4_096
    _OFFSET_BATCH_SIZE = 64
    _MAX_SURFELS_PER_CELL = 8
    _NORMAL_COSINE_MIN = math.cos(math.radians(30.0))
    _THICKNESS_WEIGHT = 0.25
    _RING_RADII = (1, 2, 4, 6)

    def __init__(
        self,
        priors: SurfacePriorData,
        *,
        device: torch.device,
        voxel_size: float,
    ) -> None:
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("LiDAR geometry voxel size must be finite and positive")

        source_points = np.ascontiguousarray(priors.points_cpu, dtype=np.float32)
        centroids = np.ascontiguousarray(
            priors.centroids.detach().float().cpu().numpy()
        )
        normals = np.ascontiguousarray(
            priors.normals.detach().float().cpu().numpy()
        )
        confidence = np.ascontiguousarray(
            priors.confidence.detach().float().cpu().numpy()
        )
        radius = np.ascontiguousarray(
            priors.radius.detach().float().cpu().numpy()
        )
        local_scale = np.ascontiguousarray(
            priors.local_scale.detach().float().cpu().numpy()
        )
        normal_scale = np.ascontiguousarray(
            priors.normal_scale.detach().float().cpu().numpy()
        )

        reliable = (
            np.isfinite(source_points).all(axis=1)
            & np.isfinite(centroids).all(axis=1)
            & np.isfinite(normals).all(axis=1)
            & np.isfinite(confidence)
            & np.isfinite(radius)
            & np.isfinite(local_scale)
            & np.isfinite(normal_scale)
            & (confidence > 0.0)
            & (radius > 0.0)
            & (local_scale > 0.0)
            & (normal_scale > 0.0)
        )
        reliable_ids = np.flatnonzero(reliable)
        if reliable_ids.size == 0:
            raise ValueError("KNN-PCA produced no reliable LiDAR surfels")

        grid_coords = np.floor(source_points[reliable_ids] / voxel_size).astype(
            np.int64
        )
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
        # Confidence-descending order within a cell makes any capacity clipping
        # deterministic and preserves its most reliable local planes.
        order = np.lexsort((-confidence[reliable_ids], linear_keys))
        sorted_keys = linear_keys[order]
        surfel_source_ids = reliable_ids[order]
        cell_keys, cell_starts, cell_counts = np.unique(
            sorted_keys, return_index=True, return_counts=True
        )
        cell_capacity = min(
            int(cell_counts.max()), self._MAX_SURFELS_PER_CELL
        )
        cell_surfel_ids = np.full(
            (len(cell_keys), cell_capacity), -1, dtype=np.int32
        )
        for slot in range(cell_capacity):
            has_slot = cell_counts > slot
            cell_surfel_ids[has_slot, slot] = (
                cell_starts[has_slot] + slot
            ).astype(np.int32)

        tensor_kwargs = {"device": device}
        self._cell_keys = torch.from_numpy(cell_keys).to(
            dtype=torch.int64, **tensor_kwargs
        )
        self._cell_surfel_ids = torch.from_numpy(cell_surfel_ids).to(
            dtype=torch.int32, **tensor_kwargs
        )
        self._points = torch.from_numpy(centroids[surfel_source_ids]).to(
            dtype=torch.float32, **tensor_kwargs
        )
        self._normals = F.normalize(
            torch.from_numpy(normals[surfel_source_ids]).to(
                dtype=torch.float32, **tensor_kwargs
            ),
            dim=-1,
            eps=1e-12,
        )
        self._confidence = torch.from_numpy(confidence[surfel_source_ids]).to(
            dtype=torch.float32, **tensor_kwargs
        )
        self._radius = torch.from_numpy(radius[surfel_source_ids]).to(
            dtype=torch.float32, **tensor_kwargs
        )
        self._local_scale = torch.from_numpy(local_scale[surfel_source_ids]).to(
            dtype=torch.float32, **tensor_kwargs
        )
        self._normal_scale = torch.from_numpy(
            normal_scale[surfel_source_ids]
        ).to(dtype=torch.float32, **tensor_kwargs)
        self._grid_min = torch.from_numpy(grid_min).to(
            dtype=torch.int64, **tensor_kwargs
        )
        self._grid_dims = torch.from_numpy(grid_dims).to(
            dtype=torch.int64, **tensor_kwargs
        )
        self._offset_shells = self._build_offset_shells(device)

        self.voxel_size = float(voxel_size)
        self.num_surfels = len(surfel_source_ids)
        self.num_cells = len(cell_keys)
        self.max_cell_occupancy = int(cell_counts.max())
        self.num_clipped_surfels = int(
            np.maximum(cell_counts - cell_capacity, 0).sum()
        )

    @classmethod
    def _build_offset_shells(cls, device: torch.device) -> Tuple[Tensor, ...]:
        shells = []
        previous_radius = -1
        for radius in cls._RING_RADII:
            offsets = [
                (x, y, z)
                for x in range(-radius, radius + 1)
                for y in range(-radius, radius + 1)
                for z in range(-radius, radius + 1)
                if max(abs(x), abs(y), abs(z)) > previous_radius
            ]
            shells.append(torch.tensor(offsets, dtype=torch.int64, device=device))
            previous_radius = radius
        return tuple(shells)

    def should_compute(self, step: int) -> bool:
        return step % self.update_every == 0

    def warmup_factor(self, step: int) -> float:
        return min((step + 1) / self.warmup_steps, 1.0)

    def _sample_visible_ids(self, visible: Tensor, *, step: int) -> Tensor:
        ids = torch.where(visible)[0]
        if ids.numel() <= self._MAX_LOSS_GAUSSIANS:
            return ids
        stride = (ids.numel() + self._MAX_LOSS_GAUSSIANS - 1) // (
            self._MAX_LOSS_GAUSSIANS
        )
        offset = (step // self.update_every) % stride
        return ids[offset::stride][: self._MAX_LOSS_GAUSSIANS]

    def _merge_offset_batch(
        self,
        means: Tensor,
        query_coords: Tensor,
        query_ids: Tensor,
        offsets: Tensor,
        top_distances: Tensor,
        top_ids: Tensor,
    ) -> None:
        coords = query_coords[query_ids, None, :] + offsets[None, :, :]
        in_grid = ((coords >= 0) & (coords < self._grid_dims)).all(dim=-1)
        keys = (
            (coords[..., 0] * self._grid_dims[1] + coords[..., 1])
            * self._grid_dims[2]
            + coords[..., 2]
        ).contiguous()

        slots = torch.searchsorted(self._cell_keys, keys)
        safe_slots = slots.clamp_max(len(self._cell_keys) - 1)
        cell_matched = (
            in_grid
            & (slots < len(self._cell_keys))
            & (self._cell_keys[safe_slots] == keys)
        )
        candidate_ids = self._cell_surfel_ids[safe_slots]
        candidate_valid = cell_matched[..., None] & (candidate_ids >= 0)
        safe_ids = candidate_ids.clamp_min(0).to(torch.long)
        candidate_points = self._points[safe_ids]
        candidate_distances = (
            candidate_points - means[query_ids, None, None, :]
        ).square().sum(dim=-1)
        candidate_distances.masked_fill_(~candidate_valid, torch.inf)

        current_distances = top_distances[query_ids]
        current_ids = top_ids[query_ids]
        combined_distances = torch.cat(
            (current_distances, candidate_distances.flatten(1)), dim=1
        )
        combined_ids = torch.cat(
            (current_ids, safe_ids.flatten(1)), dim=1
        )
        nearest_distances, nearest_slots = torch.topk(
            combined_distances,
            k=self._NUM_PLANES,
            dim=1,
            largest=False,
            sorted=True,
        )
        top_distances[query_ids] = nearest_distances
        top_ids[query_ids] = combined_ids.gather(1, nearest_slots)

    def _query_topk_chunk(self, means: Tensor) -> Tuple[Tensor, Tensor]:
        query_coords = torch.floor(means / self.voxel_size).to(torch.int64)
        query_coords = query_coords - self._grid_min
        top_distances = torch.full(
            (len(means), self._NUM_PLANES),
            torch.inf,
            dtype=means.dtype,
            device=means.device,
        )
        top_ids = torch.zeros(
            (len(means), self._NUM_PLANES),
            dtype=torch.long,
            device=means.device,
        )
        query_ids = torch.arange(len(means), device=means.device)

        for offsets in self._offset_shells:
            if query_ids.numel() == 0:
                break
            for start in range(0, len(offsets), self._OFFSET_BATCH_SIZE):
                self._merge_offset_batch(
                    means,
                    query_coords,
                    query_ids,
                    offsets[start : start + self._OFFSET_BATCH_SIZE],
                    top_distances,
                    top_ids,
                )
            has_enough = torch.isfinite(top_distances[query_ids]).sum(dim=1) >= (
                self._NUM_PLANES
            )
            query_ids = query_ids[~has_enough]

        return top_ids, top_distances

    def _query_nearby_surfels(self, means: Tensor) -> Tuple[Tensor, Tensor]:
        ids, distances = [], []
        detached_means = means.detach()
        for start in range(0, len(detached_means), self._QUERY_CHUNK_SIZE):
            chunk_ids, chunk_distances = self._query_topk_chunk(
                detached_means[start : start + self._QUERY_CHUNK_SIZE]
            )
            ids.append(chunk_ids)
            distances.append(chunk_distances)
        return torch.cat(ids, dim=0), torch.cat(distances, dim=0)

    def compute_loss(
        self,
        splats: Dict[str, Tensor],
        visible: Tensor,
        *,
        step: int,
        compute_stats: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Constrain Gaussian centers and excess normal thickness to LiDAR planes."""
        means_all = splats["means"]
        if visible.shape != means_all.shape[:1]:
            raise ValueError("Visible mask must match the Gaussian count")

        sample_ids = self._sample_visible_ids(visible, step=step)
        zero = means_all.sum() * 0.0
        if sample_ids.numel() == 0:
            return zero, {
                "valid_ratio": zero.detach(),
                "sample_ratio": zero.detach(),
                "visible_count": zero.detach(),
                "sample_count": zero.detach(),
                "matched_count": zero.detach(),
            }

        means = means_all[sample_ids]
        surfel_ids, distances_sq = self._query_nearby_surfels(means)
        finite = torch.isfinite(distances_sq)
        points = self._points[surfel_ids]
        normals = self._normals[surfel_ids]
        confidence = self._confidence[surfel_ids]
        radius = self._radius[surfel_ids]
        local_scale = self._local_scale[surfel_ids]
        normal_scale = self._normal_scale[surfel_ids]

        distances = torch.sqrt(distances_sq.clamp_min(0.0))
        match_radius = (1.5 * radius).clamp(
            min=2.0 * self.voxel_size,
            max=self._RING_RADII[-1] * self.voxel_size,
        )
        valid = finite & (distances <= match_radius)
        first_valid = valid.to(torch.int64).argmax(dim=1)
        reference_normals = normals.gather(
            1, first_valid[:, None, None].expand(-1, 1, 3)
        ).squeeze(1)
        normal_compatible = (
            (normals * reference_normals[:, None, :]).sum(dim=-1).abs()
            >= self._NORMAL_COSINE_MIN
        )
        valid &= normal_compatible

        bandwidth = radius.clamp_min(self.voxel_size)
        association = confidence * torch.exp(
            -distances_sq / (2.0 * bandwidth.square())
        )
        association = association * valid.to(association.dtype)
        association_sum = association.sum(dim=1)
        matched = association_sum > 0.0
        normalized_association = association / association_sum[:, None].clamp_min(
            1e-8
        )
        association_confidence = association.max(dim=1).values

        plane_distance = (
            (means[:, None, :] - points) * normals
        ).sum(dim=-1)
        sigma = local_scale.clamp(
            min=0.5 * self.voxel_size,
            max=2.0 * self.voxel_size,
        )
        center_loss = F.smooth_l1_loss(
            plane_distance / sigma,
            torch.zeros_like(plane_distance),
            beta=1.0,
            reduction="none",
        )

        rotations = normalized_quat_to_rotmat(
            F.normalize(splats["quats"][sample_ids], dim=-1, eps=1e-12)
        )
        scales = torch.exp(splats["scales"][sample_ids])
        normal_in_gaussian_axes = torch.einsum(
            "mab,mka->mkb", rotations, normals
        )
        normal_variance = (
            normal_in_gaussian_axes.square() * scales[:, None, :].square()
        ).sum(dim=-1)
        thickness = torch.sqrt(normal_variance + 1e-12)
        excess_thickness = (thickness - normal_scale).clamp_min(0.0)
        thickness_loss = F.smooth_l1_loss(
            excess_thickness / sigma,
            torch.zeros_like(excess_thickness),
            beta=1.0,
            reduction="none",
        )

        per_gaussian = (
            normalized_association
            * (center_loss + self._THICKNESS_WEIGHT * thickness_loss)
        ).sum(dim=1)
        loss = (
            per_gaussian * association_confidence
        ).sum() / association_confidence.sum().clamp_min(1e-8)

        visible_count = visible.sum()
        sample_count = sample_ids.new_tensor(sample_ids.numel())
        sample_ratio = sample_count.to(means.dtype) / visible_count.clamp_min(1).to(
            means.dtype
        )
        stats: Dict[str, Tensor] = {
            "valid_ratio": matched.float().mean().detach(),
            "sample_ratio": sample_ratio,
            "visible_count": visible_count.detach(),
            "sample_count": sample_count.detach(),
            "matched_count": matched.sum().detach(),
            "planes_per_gaussian": valid.float().sum(dim=1).mean().detach(),
        }
        if compute_stats and matched.any():
            center = (
                normalized_association * plane_distance.detach().abs()
            ).sum(dim=1)[matched]
            normal_thickness = (
                normalized_association * thickness.detach()
            ).sum(dim=1)[matched]
            stats.update(
                {
                    "center_p50": torch.quantile(center, 0.50),
                    "center_p90": torch.quantile(center, 0.90),
                    "thickness_p90": torch.quantile(normal_thickness, 0.90),
                }
            )
        return loss, stats
