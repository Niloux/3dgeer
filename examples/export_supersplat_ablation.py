#!/usr/bin/env python3
"""Export local Gaussian-geometry ablations for inspection in SuperSplat.

The tool deliberately operates on a saved checkpoint instead of changing the
training result in place.  It writes four PLY files for the same cylindrical
ROI: an untouched baseline, scale-ratio clamping only, LiDAR alignment only,
and an aggressive cleaned combination of both.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.neighbors import NearestNeighbors


_LIDAR_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def _read_lidar_xyz(path: Path, transform: Optional[np.ndarray] = None) -> np.ndarray:
    """Read the RGB LiDAR PLY layout used by ``simple_trainer``."""
    if not path.is_file():
        raise FileNotFoundError(f"LiDAR PLY not found: {path}")
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        count_line = next(
            (line for line in header if line.startswith("element vertex ")), None
        )
        if count_line is None or "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Unsupported LiDAR PLY: {path}")
        count = int(count_line.split()[2])
        expected = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        properties = [line for line in header if line.startswith("property ")]
        if properties != expected:
            raise ValueError(f"Unsupported LiDAR PLY properties: {path}")
        vertices = np.fromfile(stream, dtype=_LIDAR_DTYPE, count=count)
        if len(vertices) != count or stream.read(1):
            raise ValueError(f"Invalid LiDAR PLY payload: {path}")

    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    if transform is not None:
        transform = np.asarray(transform, dtype=np.float64)
        points = points @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(points).all():
        raise ValueError("LiDAR contains non-finite coordinates")
    return np.ascontiguousarray(points, dtype=np.float32)


def quat_to_rotmat(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized-or-unnormalized wxyz quaternions to matrices."""
    q = np.asarray(quaternions, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - w * z)
    matrices[:, 0, 2] = 2.0 * (x * z + w * y)
    matrices[:, 1, 0] = 2.0 * (x * y + w * z)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - w * x)
    matrices[:, 2, 0] = 2.0 * (x * z - w * y)
    matrices[:, 2, 1] = 2.0 * (y * z + w * x)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def rotmat_to_quat(matrices: np.ndarray) -> np.ndarray:
    """Convert right-handed rotation matrices to normalized wxyz quaternions."""
    m = np.asarray(matrices, dtype=np.float64)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    q_abs = np.sqrt(
        np.maximum(
            np.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                axis=1,
            ),
            0.0,
        )
    )
    numerators = np.stack(
        (
            np.stack((q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01), axis=1),
            np.stack((m21 - m12, q_abs[:, 1] ** 2, m10 + m01, m02 + m20), axis=1),
            np.stack((m02 - m20, m10 + m01, q_abs[:, 2] ** 2, m12 + m21), axis=1),
            np.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[:, 3] ** 2), axis=1),
        ),
        axis=1,
    )
    candidates = numerators / np.maximum(2.0 * q_abs[:, :, None], 1e-12)
    best = np.argmax(q_abs, axis=1)
    quaternions = candidates[np.arange(len(m)), best]
    quaternions /= np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12
    )
    return quaternions.astype(np.float32)


def clamp_log_scales(
    log_scales: np.ndarray,
    *,
    min_to_mid: float = 0.15,
    max_to_mid: float = 4.0,
) -> np.ndarray:
    """Clamp the smallest/middle and largest/middle actual-scale ratios."""
    if not 0.0 < min_to_mid <= 1.0:
        raise ValueError("min_to_mid must be in (0, 1]")
    if max_to_mid < 1.0:
        raise ValueError("max_to_mid must be >= 1")
    actual = np.exp(np.asarray(log_scales, dtype=np.float64))
    order = np.argsort(actual, axis=1)
    ordered = np.take_along_axis(actual, order, axis=1)
    ordered[:, 0] = np.maximum(ordered[:, 0], min_to_mid * ordered[:, 1])
    ordered[:, 2] = np.minimum(ordered[:, 2], max_to_mid * ordered[:, 1])
    clamped = np.empty_like(ordered)
    np.put_along_axis(clamped, order, ordered, axis=1)
    return np.log(np.maximum(clamped, 1e-12)).astype(np.float32)


def align_min_axes_to_normals(
    quaternions: np.ndarray,
    log_scales: np.ndarray,
    target_normals: np.ndarray,
) -> np.ndarray:
    """Rigidly rotate each Gaussian's smallest-scale axis onto a target normal."""
    matrices = quat_to_rotmat(quaternions)
    min_axes = np.argmin(log_scales, axis=1)
    source = matrices[np.arange(len(matrices)), :, min_axes]
    target = np.asarray(target_normals, dtype=np.float64).copy()
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    target[np.sum(source * target, axis=1) < 0.0] *= -1.0

    cross = np.cross(source, target)
    dot = np.clip(np.sum(source * target, axis=1), -1.0, 1.0)
    skew = np.zeros_like(matrices)
    skew[:, 0, 1] = -cross[:, 2]
    skew[:, 0, 2] = cross[:, 1]
    skew[:, 1, 0] = cross[:, 2]
    skew[:, 1, 2] = -cross[:, 0]
    skew[:, 2, 0] = -cross[:, 1]
    skew[:, 2, 1] = cross[:, 0]
    identity = np.broadcast_to(np.eye(3), matrices.shape)
    align = identity + skew + (skew @ skew) / np.maximum(
        (1.0 + dot)[:, None, None], 1e-12
    )
    return rotmat_to_quat(align @ matrices)


def estimate_ground_height(
    lidar: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
    camera_height: float,
    *,
    bin_size: float = 0.05,
) -> float:
    """Estimate the dominant low horizontal surface from a local z histogram."""
    radial = np.linalg.norm(lidar[:, :2] - center_xy[None], axis=1)
    z = lidar[(radial <= radius) & (lidar[:, 2] < camera_height - 0.4), 2]
    if len(z) < 100:
        raise ValueError("Too few local LiDAR samples to estimate the ground height")
    lower = math.floor(float(z.min()) / bin_size) * bin_size
    upper = math.ceil(float(z.max()) / bin_size) * bin_size + bin_size
    hist, edges = np.histogram(z, bins=np.arange(lower, upper + 0.5 * bin_size, bin_size))
    peak = int(np.argmax(hist))
    in_peak = z[(z >= edges[peak]) & (z < edges[peak + 1])]
    return float(np.median(in_peak))


def query_surface_planes(
    query_points: np.ndarray,
    lidar_points: np.ndarray,
    *,
    k: int,
    planarity_threshold: float,
    curvature_threshold: float,
    max_distance: float,
    min_confidence: float,
    max_normal_angle_deg: float,
    chunk_size: int = 32_768,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local plane centroids, normals, confidence, and validity."""
    if len(lidar_points) < 4:
        raise ValueError("At least four LiDAR points are required for surface queries")
    k_eff = min(k, len(lidar_points))
    neighbors_model = NearestNeighbors(
        n_neighbors=k_eff, metric="euclidean", n_jobs=-1
    ).fit(np.ascontiguousarray(lidar_points, dtype=np.float32))
    centroids = np.empty_like(query_points, dtype=np.float32)
    normals = np.empty_like(query_points, dtype=np.float32)
    confidence = np.zeros(len(query_points), dtype=np.float32)
    nearest = np.full(len(query_points), np.inf, dtype=np.float32)

    for start in range(0, len(query_points), chunk_size):
        end = min(start + chunk_size, len(query_points))
        distances, indices = neighbors_model.kneighbors(query_points[start:end])
        neighbors = lidar_points[indices]
        centroid = neighbors.mean(axis=1)
        centered = neighbors - centroid[:, None]
        covariance = centered.transpose(0, 2, 1) @ centered / float(k_eff - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        lambda0, lambda1, lambda2 = eigenvalues.T
        planarity = (lambda1 - lambda0) / np.maximum(lambda2, 1e-12)
        curvature = lambda0 / np.maximum(eigenvalues.sum(axis=1), 1e-12)
        planar = (planarity >= planarity_threshold) & (
            curvature <= curvature_threshold
        )
        planarity_score = np.clip(
            (planarity - planarity_threshold) / max(1.0 - planarity_threshold, 1e-12),
            0.0,
            1.0,
        )
        if curvature_threshold > 0.0:
            curvature_score = np.clip(1.0 - curvature / curvature_threshold, 0.0, 1.0)
        else:
            curvature_score = (curvature <= 0.0).astype(np.float32)
        centroids[start:end] = centroid
        normals[start:end] = eigenvectors[:, :, 0]
        confidence[start:end] = np.where(
            planar, planarity_score * curvature_score, 0.0
        )
        nearest[start:end] = distances[:, 0]

    vertical = np.abs(normals[:, 2]) >= math.cos(math.radians(max_normal_angle_deg))
    valid = (
        (nearest <= max_distance)
        & (confidence >= min_confidence)
        & vertical
    )
    return centroids, normals, confidence, valid


def density_keep_mask(
    xy: np.ndarray,
    priorities: np.ndarray,
    *,
    cell_size: float,
    max_per_cell: int,
) -> np.ndarray:
    """Keep at most ``max_per_cell`` samples, preferring larger priorities."""
    if cell_size <= 0.0 or max_per_cell < 1:
        raise ValueError("Density cell size and max_per_cell must be positive")
    cells = np.floor(xy / cell_size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.lexsort((-priorities, inverse))
    ordered_groups = inverse[order]
    starts = np.flatnonzero(np.r_[True, ordered_groups[1:] != ordered_groups[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    ranks = np.arange(len(order)) - np.repeat(starts, counts)
    keep = np.zeros(len(xy), dtype=bool)
    keep[order[ranks < max_per_cell]] = True
    return keep


def _iter_indices(indices: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), chunk_size):
        yield indices[start : start + chunk_size]


def write_gaussian_ply(
    path: Path,
    splats: Dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    overwrite: bool = False,
    chunk_size: int = 131_072,
) -> None:
    """Write the same uncompressed binary PLY schema as ``gsplat.export_splats``."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    indices = np.asarray(indices, dtype=np.int64)
    sh_order = splats["shN"].shape[1]
    properties = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    properties.extend(f"f_rest_{i}" for i in range(3 * sh_order))
    properties.extend(
        [
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
    )
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {len(indices)}"]
        + [f"property float {name}" for name in properties]
        + ["end_header", ""]
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(header)
            for selected in _iter_indices(indices, chunk_size):
                sh0 = splats["sh0"][selected, 0, :]
                shN = splats["shN"][selected].transpose(0, 2, 1).reshape(
                    len(selected), -1
                )
                data = np.concatenate(
                    (
                        splats["means"][selected],
                        sh0,
                        shN,
                        splats["opacities"][selected, None],
                        splats["scales"][selected],
                        splats["quats"][selected],
                    ),
                    axis=1,
                ).astype("<f4", copy=False)
                if not np.isfinite(data).all():
                    raise ValueError(f"Cannot export non-finite values to {path}")
                stream.write(data.tobytes())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_checkpoint(path: Path) -> Dict[str, np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    missing = required.difference(checkpoint["splats"])
    if missing:
        raise KeyError(f"Checkpoint is missing splat fields: {sorted(missing)}")
    return {
        name: tensor.detach().contiguous().numpy()
        for name, tensor in checkpoint["splats"].items()
        if name in required
    }


def _latest_step(result_dir: Path) -> int:
    candidates = []
    for path in (result_dir / "ckpts").glob("ckpt_*_rank0.pt"):
        try:
            candidates.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    if not candidates:
        raise FileNotFoundError(f"No rank-0 checkpoints found in {result_dir / 'ckpts'}")
    return max(candidates)


def _camera_context(cfg: Dict[str, Any]) -> Tuple[np.ndarray, float, np.ndarray]:
    from datasets.colmap import Parser

    parser = Parser(
        data_dir=cfg["data_dir"],
        factor=cfg.get("data_factor", 1),
        normalize=cfg.get("normalize_world_space", False),
        test_every=cfg.get("test_every", 8),
        use_test_split=cfg.get("use_test_split", True),
        undistort=not cfg.get("keep_distortion", False),
        max_fisheye_fov=cfg.get("max_fisheye_fov"),
        frame_id_min=cfg.get("frame_id_min"),
        frame_id_max=cfg.get("frame_id_max"),
        sky_mask_dir=cfg.get("sky_mask_dir") if cfg.get("sky_enabled") else None,
    )
    centers = parser.camtoworlds[:, :3, 3]
    median = np.median(centers, axis=0)
    return median[:2].astype(np.float32), float(median[2]), parser.transform


def _ratio_stats(log_scales: np.ndarray) -> Dict[str, float]:
    actual = np.sort(np.exp(log_scales.astype(np.float64)), axis=1)
    mid = np.maximum(actual[:, 1], 1e-12)
    return {
        "min_mid_q50": float(np.quantile(actual[:, 0] / mid, 0.5)),
        "min_mid_q10": float(np.quantile(actual[:, 0] / mid, 0.1)),
        "max_mid_q50": float(np.quantile(actual[:, 2] / mid, 0.5)),
        "max_mid_q90": float(np.quantile(actual[:, 2] / mid, 0.9)),
        "max_mid_q99": float(np.quantile(actual[:, 2] / mid, 0.99)),
    }


def export_ablation(args: argparse.Namespace) -> Dict[str, Any]:
    result_dir = args.result_dir.resolve()
    cfg_path = result_dir / "cfg.yml"
    with cfg_path.open() as stream:
        cfg = yaml.safe_load(stream)
    step = args.step if args.step is not None else _latest_step(result_dir)
    checkpoint_path = result_dir / "ckpts" / f"ckpt_{step}_rank0.pt"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_dir / "supersplat_ablation"
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    splats = _load_checkpoint(checkpoint_path)
    inferred_center, camera_z, transform = _camera_context(cfg)
    center_xy = (
        np.asarray(args.center, dtype=np.float32)
        if args.center is not None
        else inferred_center
    )
    lidar_path = Path(cfg.get("init_lidar_path") or Path(cfg["data_dir"]) / "lidar.ply")
    lidar = _read_lidar_xyz(lidar_path, transform=transform)
    ground_z = (
        args.ground_z
        if args.ground_z is not None
        else estimate_ground_height(lidar, center_xy, args.roi_radius, camera_z)
    )

    means = splats["means"]
    radial = np.linalg.norm(means[:, :2] - center_xy[None], axis=1)
    export_mask = np.ones(len(means), dtype=bool) if args.full_scene else radial <= args.roi_radius
    export_indices = np.flatnonzero(export_mask)
    # ``--full-scene`` changes only what is exported.  Geometry surgery remains
    # local to the diagnostic ROI so a local LiDAR model is never extrapolated
    # across the whole scene.
    ground_mask = (radial <= args.roi_radius) & (
        np.abs(means[:, 2] - ground_z) <= args.ground_half_height
    )
    ground_indices = np.flatnonzero(ground_mask)
    local_lidar_mask = (
        np.linalg.norm(lidar[:, :2] - center_xy[None], axis=1)
        <= args.roi_radius + 1.0
    ) & (np.abs(lidar[:, 2] - ground_z) <= args.lidar_ground_half_height)
    ground_lidar = lidar[local_lidar_mask]
    print(
        f"ROI center=({center_xy[0]:.3f}, {center_xy[1]:.3f}), "
        f"ground_z={ground_z:.3f}; exporting {len(export_indices):,} splats, "
        f"testing {len(ground_indices):,} ground splats against {len(ground_lidar):,} LiDAR points"
    )

    centroids, normals, confidence, valid = query_surface_planes(
        means[ground_indices],
        ground_lidar,
        k=args.knn_k,
        planarity_threshold=args.planarity_threshold,
        curvature_threshold=args.curvature_threshold,
        max_distance=args.max_surface_distance,
        min_confidence=args.min_surface_confidence,
        max_normal_angle_deg=args.max_normal_angle_deg,
    )
    valid_indices = ground_indices[valid]
    signed_distance = np.sum(
        (means[valid_indices] - centroids[valid]) * normals[valid], axis=1
    )

    suffix = "full" if args.full_scene else "roi"
    files: Dict[str, Dict[str, Any]] = {}

    def write_variant(name: str, data: Dict[str, np.ndarray], indices: np.ndarray) -> None:
        path = output_dir / f"{name}_{suffix}.ply"
        print(f"Writing {path.name}: {len(indices):,} splats")
        write_gaussian_ply(path, data, indices, overwrite=args.overwrite)
        files[name] = {"path": str(path), "count": int(len(indices))}

    write_variant("baseline", splats, export_indices)

    shape_clamped = dict(splats)
    shape_clamped["scales"] = splats["scales"].copy()
    shape_clamped["scales"][ground_indices] = clamp_log_scales(
        splats["scales"][ground_indices],
        min_to_mid=args.min_to_mid,
        max_to_mid=args.max_to_mid,
    )
    write_variant("shape_clamped", shape_clamped, export_indices)

    surface_aligned = dict(splats)
    surface_aligned["means"] = splats["means"].copy()
    surface_aligned["quats"] = splats["quats"].copy()
    surface_aligned["means"][valid_indices] -= signed_distance[:, None] * normals[valid]
    surface_aligned["quats"][valid_indices] = align_min_axes_to_normals(
        splats["quats"][valid_indices], splats["scales"][valid_indices], normals[valid]
    )
    write_variant("surface_aligned", surface_aligned, export_indices)

    opacity = 1.0 / (1.0 + np.exp(-splats["opacities"][ground_indices]))
    eligible = valid & (opacity >= args.min_opacity)
    density = np.zeros(len(ground_indices), dtype=bool)
    density[eligible] = density_keep_mask(
        means[ground_indices[eligible], :2],
        opacity[eligible],
        cell_size=args.density_cell_size,
        max_per_cell=args.max_per_cell,
    )
    cleaned_keep = export_mask.copy()
    cleaned_keep[ground_indices] = density
    cleaned_indices = np.flatnonzero(cleaned_keep)
    surface_cleaned = dict(surface_aligned)
    surface_cleaned["scales"] = splats["scales"].copy()
    kept_ground_indices = ground_indices[density]
    surface_cleaned["scales"][kept_ground_indices] = clamp_log_scales(
        splats["scales"][kept_ground_indices],
        min_to_mid=args.min_to_mid,
        max_to_mid=args.max_to_mid,
    )
    write_variant("surface_cleaned", surface_cleaned, cleaned_indices)

    manifest: Dict[str, Any] = {
        "source_checkpoint": str(checkpoint_path),
        "step": step,
        "roi": {
            "full_scene": args.full_scene,
            "center_xy": center_xy.tolist(),
            "radius": args.roi_radius,
            "ground_z": ground_z,
            "ground_half_height": args.ground_half_height,
        },
        "thresholds": {
            "knn_k": args.knn_k,
            "planarity": args.planarity_threshold,
            "curvature": args.curvature_threshold,
            "surface_distance": args.max_surface_distance,
            "surface_confidence": args.min_surface_confidence,
            "normal_angle_deg": args.max_normal_angle_deg,
            "min_to_mid": args.min_to_mid,
            "max_to_mid": args.max_to_mid,
            "min_opacity": args.min_opacity,
            "density_cell_size": args.density_cell_size,
            "max_per_cell": args.max_per_cell,
        },
        "counts": {
            "source": int(len(means)),
            "export_roi": int(len(export_indices)),
            "ground_candidates": int(len(ground_indices)),
            "valid_surface": int(valid.sum()),
            "eligible_after_opacity": int(eligible.sum()),
            "cleaned_ground": int(density.sum()),
        },
        "ground_scale_ratios_before": _ratio_stats(splats["scales"][ground_indices]),
        "ground_scale_ratios_clamped": _ratio_stats(
            shape_clamped["scales"][ground_indices]
        ),
        "files": files,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_{suffix}.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dir", type=Path, help="Training result containing cfg.yml and ckpts/"
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Checkpoint step; defaults to latest rank-0 checkpoint",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--roi-radius", type=float, default=8.0)
    parser.add_argument("--full-scene", action="store_true")
    parser.add_argument("--ground-z", type=float)
    parser.add_argument("--ground-half-height", type=float, default=0.35)
    parser.add_argument("--lidar-ground-half-height", type=float, default=0.45)
    parser.add_argument("--knn-k", type=int, default=24)
    parser.add_argument("--planarity-threshold", type=float, default=0.30)
    parser.add_argument("--curvature-threshold", type=float, default=0.10)
    parser.add_argument("--max-surface-distance", type=float, default=0.20)
    parser.add_argument("--min-surface-confidence", type=float, default=0.25)
    parser.add_argument("--max-normal-angle-deg", type=float, default=35.0)
    parser.add_argument("--min-to-mid", type=float, default=0.15)
    parser.add_argument("--max-to-mid", type=float, default=4.0)
    parser.add_argument("--min-opacity", type=float, default=0.10)
    parser.add_argument("--density-cell-size", type=float, default=0.05)
    parser.add_argument("--max-per-cell", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    export_ablation(build_arg_parser().parse_args())
