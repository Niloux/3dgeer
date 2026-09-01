import math
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from utils import rgb_to_sh


_SH_C0 = 0.28209479177387814
_SKY_COLOR_MIN = 1.0 / 255.0


def rotation_matrix_to_quaternion(rotation: Tensor) -> Tensor:
    """Convert right-handed rotation matrices to normalized wxyz quaternions."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation matrices with shape [..., 3, 3], got {rotation.shape}"
        )

    m00, m01, m02 = rotation[..., 0, 0], rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m11, m12 = rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 1, 2]
    m20, m21, m22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp_min(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            0.0,
        )
    )
    quat_by_component = torch.stack(
        (
            torch.stack(
                (q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01),
                dim=-1,
            ),
            torch.stack(
                (m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20),
                dim=-1,
            ),
            torch.stack(
                (m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21),
                dim=-1,
            ),
            torch.stack(
                (m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3].square()),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    candidates = quat_by_component / (2.0 * q_abs[..., None].clamp_min(1e-8))
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(*best.shape, 1, 4)
    quaternion = torch.gather(candidates, dim=-2, index=gather_index).squeeze(-2)
    return torch.nn.functional.normalize(quaternion, dim=-1)


@torch.no_grad()
def initialize_surface_priors_knn_pca(
    points: Tensor,
    *,
    k: int,
    local_scale_factor: float,
    normal_scale_factor: float,
    planarity_threshold: float,
    curvature_threshold: float,
) -> Tuple[Tensor, Tensor, int]:
    """Build surface-aligned quaternions and scales from local KNN-PCA."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    if k < 4:
        raise ValueError(f"KNN-PCA k must be >= 4, got {k}")
    if local_scale_factor <= 0.0 or normal_scale_factor <= 0.0:
        raise ValueError("KNN-PCA scale factors must be positive")
    if not 0.0 <= planarity_threshold <= 1.0:
        raise ValueError("KNN-PCA planarity threshold must be in [0, 1]")
    if not 0.0 <= curvature_threshold <= 1.0:
        raise ValueError("KNN-PCA curvature threshold must be in [0, 1]")
    if not torch.isfinite(points).all():
        raise ValueError("KNN-PCA points contain non-finite values")

    num_points = points.shape[0]
    dtype, device = points.dtype, points.device
    min_scale = 1e-6
    rotations = torch.zeros((num_points, 4), dtype=dtype, device=device)
    rotations[:, 0] = 1.0
    if num_points < 4:
        scales = torch.full(
            (num_points, 3), min_scale, dtype=dtype, device=device
        )
        return rotations, scales, 0

    k_eff = min(k, num_points)
    points_cpu = np.ascontiguousarray(points.detach().float().cpu().numpy())
    neighbors_model = NearestNeighbors(
        n_neighbors=k_eff,
        metric="euclidean",
        n_jobs=-1,
    ).fit(points_cpu)
    actual_scales = torch.empty((num_points, 3), dtype=dtype, device=device)
    accepted_count = 0
    eps = torch.finfo(dtype).eps

    for start in range(0, num_points, 65_536):
        end = min(start + 65_536, num_points)
        distances, neighbor_indices = neighbors_model.kneighbors(points_cpu[start:end])
        neighbors = points_cpu[neighbor_indices]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = centered.transpose(0, 2, 1) @ centered / float(k_eff - 1)
        if not np.isfinite(covariance).all():
            raise ValueError("KNN-PCA covariance contains non-finite values")

        # NumPy is deliberate here. Batched torch.linalg.eigh on CUDA has raised
        # CUSOLVER_STATUS_INVALID_VALUE for large, finite Park point clouds.
        eigenvalues_np, eigenvectors = np.linalg.eigh(covariance)
        rotation_matrix_np = np.stack(
            (eigenvectors[:, :, 2], eigenvectors[:, :, 1], eigenvectors[:, :, 0]),
            axis=-1,
        )
        rotation_matrix_np[
            np.linalg.det(rotation_matrix_np) < 0.0, :, 0
        ] *= -1.0
        rotation_matrix = torch.from_numpy(rotation_matrix_np).to(
            device=device, dtype=dtype
        )
        eigenvalues = torch.from_numpy(eigenvalues_np).to(device=device, dtype=dtype)
        eigenvalues = eigenvalues.clamp_min(0.0)
        lambda0, lambda1, lambda2 = eigenvalues.unbind(dim=-1)
        planarity = (lambda1 - lambda0) / lambda2.clamp_min(eps)
        curvature = lambda0 / eigenvalues.sum(dim=-1).clamp_min(eps)
        valid = (planarity >= planarity_threshold) & (
            curvature <= curvature_threshold
        )

        quaternions = rotation_matrix_to_quaternion(rotation_matrix)
        tangent_minor = torch.sqrt(lambda1.clamp_min(eps))
        tangent_major = torch.sqrt(lambda2.clamp_min(eps))
        tangent_mean = torch.sqrt((tangent_minor * tangent_major).clamp_min(eps))
        tangent_ratio_major = (tangent_major / tangent_mean).clamp(0.5, 2.0)
        tangent_ratio_minor = (tangent_minor / tangent_mean).clamp(0.5, 2.0)

        neighbor_distances = torch.from_numpy(distances[:, 1:4]).to(
            device=device, dtype=dtype
        )
        base = (
            torch.sqrt(neighbor_distances.square().mean(dim=-1))
            * local_scale_factor
        ).clamp_min(min_scale)
        isotropic_scales = base[:, None].repeat(1, 3)
        pca_scales = torch.stack(
            (
                base * tangent_ratio_major,
                base * tangent_ratio_minor,
                base * normal_scale_factor,
            ),
            dim=-1,
        ).clamp_min(min_scale)
        actual_scales[start:end] = torch.where(
            valid[:, None], pca_scales, isotropic_scales
        )
        rotations[start:end] = torch.where(
            valid[:, None], quaternions, rotations[start:end]
        )
        accepted_count += int(valid.sum().item())

    return rotations, actual_scales, accepted_count


def sky_hemisphere(
    count: int, radius: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic Fibonacci upper-hemisphere points and KNN scales."""
    if count < 2 or radius <= 0.0 or seed < 0:
        raise ValueError(
            "Gaussian sky requires at least two points, a positive radius, "
            "and a non-negative seed"
        )

    rng = np.random.default_rng(seed)
    index = np.arange(count, dtype=np.float64)
    z = (index + 0.5) / count
    phi = index * (math.pi * (3.0 - math.sqrt(5.0))) + rng.random() * math.tau
    xy = np.sqrt(1.0 - z * z)
    xyz = (
        np.column_stack((xy * np.cos(phi), xy * np.sin(phi), z))
        * np.float32(radius)
    ).astype(np.float32)

    cv2.setRNGSeed(seed % 2_147_483_647)
    flann = cv2.flann_Index(xyz, {"algorithm": 1, "trees": 8})
    _, squared_distances = flann.knnSearch(
        xyz, 2, params={"checks": min(count, 256)}
    )
    flann.release()
    minimum = np.finfo(np.float32).eps * radius
    scales = np.sqrt(np.maximum(squared_distances[:, 1], minimum**2)).astype(
        np.float32
    )
    return xyz, scales


def create_sky_splats_with_optimizers(
    *,
    count: int,
    radius: float,
    initial_opacity: float,
    sh_degree: int,
    sh0_lr: float,
    shN_lr: float,
    seed: int,
    device: str,
    world_rank: int,
    world_size: int,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Create a strategy-independent Gaussian sky with fixed geometry."""
    if not 0.0 < initial_opacity < 1.0:
        raise ValueError("sky_initial_opacity must be between zero and one")
    if not 0 <= sh_degree <= 3:
        raise ValueError("Gaussian sky supports SH degrees from 0 through 3")
    if sh0_lr < 0.0 or shN_lr < 0.0:
        raise ValueError("Gaussian sky learning rates must be non-negative")
    if world_size < 1 or not 0 <= world_rank < world_size:
        raise ValueError("Invalid distributed rank for Gaussian sky")

    points_np, scales_np = sky_hemisphere(count, radius, seed)
    points = torch.from_numpy(points_np)[world_rank::world_size]
    scales = torch.from_numpy(scales_np)[world_rank::world_size]
    local_count = points.shape[0]

    colors = torch.zeros((local_count, (sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(torch.ones((local_count, 3)))
    parameters = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points, requires_grad=False),
            "scales": torch.nn.Parameter(
                torch.log(scales.clamp_min(1e-6))[:, None].repeat(1, 3),
                requires_grad=False,
            ),
            "quats": torch.nn.Parameter(
                torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(local_count, 1),
                requires_grad=False,
            ),
            "opacities": torch.nn.Parameter(
                torch.full(
                    (local_count,), torch.logit(torch.tensor(initial_opacity)).item()
                ),
                requires_grad=False,
            ),
            "sh0": torch.nn.Parameter(colors[:, :1, :]),
            "shN": torch.nn.Parameter(colors[:, 1:, :]),
        }
    ).to(device)

    fused = str(device).startswith("cuda")
    optimizers = {
        "sh0": torch.optim.Adam(
            [{"params": parameters["sh0"], "lr": sh0_lr, "name": "sky_sh0"}],
            eps=1e-15,
            fused=fused,
        ),
        "shN": torch.optim.Adam(
            [{"params": parameters["shN"], "lr": shN_lr, "name": "sky_shN"}],
            eps=1e-15,
            fused=fused,
        ),
    }
    return parameters, optimizers


def composite_sky(
    foreground: Tensor, foreground_alpha: Tensor, sky: Tensor
) -> Tensor:
    """Alpha-composite a separately rendered sky behind the foreground."""
    if foreground.shape != sky.shape:
        raise ValueError(
            f"Foreground and sky shapes must match, got {foreground.shape} and {sky.shape}"
        )
    if foreground_alpha.shape != foreground.shape[:-1] + (1,):
        raise ValueError(
            "Foreground alpha must have shape [..., H, W, 1] matching the RGB images"
        )
    return foreground + (1.0 - foreground_alpha) * sky


@torch.no_grad()
def clamp_sky_sh_colors(splats: torch.nn.ParameterDict, sh_degree: int) -> None:
    """Keep the learned sky radiance in a stable displayable range."""
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast

    coefficients = torch.cat((splats["sh0"], splats["shN"]), dim=1)
    directions = torch.nn.functional.normalize(splats["means"], dim=-1)
    bases = _eval_sh_bases_fast((sh_degree + 1) ** 2, directions)
    colors = (bases[..., None] * coefficients).sum(dim=-2) + 0.5
    correction = (colors.clamp(_SKY_COLOR_MIN, 1.0) - colors) / _SH_C0
    splats["sh0"].add_(correction[:, None, :])
