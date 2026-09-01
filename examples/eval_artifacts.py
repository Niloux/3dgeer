"""Structured, human-readable artifacts for simple trainer evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_RGB_ERROR_MAX = 0.25
_INV_DEPTH_ERROR_MAX = 0.25
_TILE_MAX_WIDTH = 800
_LABEL_HEIGHT = 30
_HEADER_HEIGHT = 42
_GAP = 4


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@lru_cache(maxsize=None)
def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _rgb8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _heatmap(
    values: np.ndarray,
    maximum: float,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Small dependency-free black-red-yellow-white scalar colormap."""
    finite_values = np.where(np.isfinite(values), values, 0.0)
    scaled = np.clip(finite_values / maximum, 0.0, 1.0)
    red = np.clip(scaled * 3.0, 0.0, 1.0)
    green = np.clip(scaled * 3.0 - 1.0, 0.0, 1.0)
    blue = np.clip(scaled * 3.0 - 2.0, 0.0, 1.0)
    colors = _rgb8(np.stack((red, green, blue), axis=-1))
    invalid = ~np.isfinite(values)
    if valid is not None:
        invalid |= ~valid
    colors[invalid] = 0
    return colors


def _depth_visual(depth: np.ndarray, depth_max: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    return _heatmap(depth, depth_max, valid)


def _alpha_visual(
    alpha: np.ndarray,
    valid_mask: Optional[np.ndarray],
    sky_mask: Optional[np.ndarray],
) -> np.ndarray:
    colors = np.repeat(_rgb8(np.clip(alpha, 0.0, 1.0))[..., None], 3, axis=-1)
    if sky_mask is not None:
        sky = sky_mask.astype(bool)
        colors[sky] = (
            colors[sky].astype(np.float32) * 0.4
            + np.array([0.0, 80.0, 180.0], dtype=np.float32) * 0.6
        ).astype(np.uint8)
    if valid_mask is not None:
        colors[~valid_mask.astype(bool)] = np.array([80, 0, 80], dtype=np.uint8)
    return colors


def _safe_image_path(image_name: str) -> Path:
    relative = PurePosixPath(image_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        relative = PurePosixPath(relative.name)
    parts = [part for part in relative.parts if part not in ("", ".", "/")]
    if not parts:
        return Path("unnamed.jpg")
    return Path(*parts).with_suffix(".jpg")


def _tile(image: np.ndarray, label: str, width: int, height: int) -> Image.Image:
    resized = Image.fromarray(image).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    tile = Image.new("RGB", (width, height + _LABEL_HEIGHT), color=(12, 12, 12))
    tile.paste(resized, (0, _LABEL_HEIGHT))
    ImageDraw.Draw(tile).text(
        (8, 6), label, fill=(235, 235, 235), font=_font(15)
    )
    return tile


@dataclass
class EvalArtifact:
    split: str
    image_name: str
    image_path: str
    source_index: int
    split_image_id: int
    camera_id: int
    camera_model: int
    rig_frame_index: int
    target_rgb: np.ndarray
    canonical_rgb: np.ndarray
    rendered_depth: np.ndarray
    foreground_alpha: np.ndarray
    metrics: Mapping[str, float]
    intrinsics: np.ndarray
    camtoworld: np.ndarray
    valid_mask: Optional[np.ndarray] = None
    sky_mask: Optional[np.ndarray] = None
    target_depth: Optional[np.ndarray] = None
    final_rgb: Optional[np.ndarray] = None
    final_label: Optional[str] = None
    radial_coeffs: Optional[np.ndarray] = None
    tangential_coeffs: Optional[np.ndarray] = None


class EvalArtifactWriter:
    """Own eval artifact naming, panel composition, and the frame manifest."""

    def __init__(
        self,
        render_dir: Path,
        iteration: int,
        completed_step: int,
        variant: str,
        depth_max: float,
        scene_scale: float,
    ) -> None:
        self.render_dir = render_dir
        self.iteration = int(iteration)
        self.completed_step = int(completed_step)
        self.variant = variant
        self.depth_max = max(float(depth_max), 1e-8)
        self.scene_scale = float(scene_scale)
        self.event_dir = (
            render_dir / f"step_{self.completed_step:06d}" / self.variant
        )
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        self._panel_sources: Dict[Path, str] = {}

    def _panel_path(self, artifact: EvalArtifact) -> Path:
        relative = Path(artifact.split) / _safe_image_path(artifact.image_name)
        previous_source = self._panel_sources.get(relative)
        if previous_source is not None and previous_source != artifact.image_name:
            relative = relative.with_name(
                f"{relative.stem}__{artifact.source_index:06d}{relative.suffix}"
            )
        self._panel_sources[relative] = artifact.image_name
        return self.event_dir / relative

    def write(self, artifact: EvalArtifact) -> None:
        target = np.asarray(artifact.target_rgb, dtype=np.float32)
        canonical = np.asarray(artifact.canonical_rgb, dtype=np.float32)
        final = (
            np.asarray(artifact.final_rgb, dtype=np.float32)
            if artifact.final_rgb is not None
            else canonical
        )
        valid_mask = (
            np.asarray(artifact.valid_mask, dtype=bool)
            if artifact.valid_mask is not None
            else None
        )
        sky_mask = (
            np.asarray(artifact.sky_mask, dtype=bool)
            if artifact.sky_mask is not None
            else None
        )

        rgb_error = np.mean(np.abs(final - target), axis=-1)
        rows = [
            [
                (_rgb8(target), "GT RGB"),
                (_rgb8(canonical), "Canonical RGB"),
            ]
        ]
        if artifact.final_rgb is not None:
            rows[0].append(
                (_rgb8(final), f"{artifact.final_label or 'Final'} RGB")
            )
        rows[0].append(
            (
                _heatmap(rgb_error, _RGB_ERROR_MAX, valid_mask),
                f"RGB abs error (0-{_RGB_ERROR_MAX:g})",
            )
        )

        rendered_depth = np.asarray(artifact.rendered_depth, dtype=np.float32)
        depth_row = [
            (
                _depth_visual(rendered_depth, self.depth_max),
                f"Rendered expected depth (0-{self.depth_max:g})",
            )
        ]
        if artifact.target_depth is not None:
            target_depth = np.asarray(artifact.target_depth, dtype=np.float32)
            depth_valid = np.isfinite(target_depth) & (target_depth > 0.0)
            if valid_mask is not None:
                depth_valid &= valid_mask
            rendered_disp = np.where(
                rendered_depth > 0.0,
                1.0 / np.maximum(rendered_depth, 1e-8),
                0.0,
            )
            target_disp = np.where(
                depth_valid,
                1.0 / np.maximum(target_depth, 1e-8),
                0.0,
            )
            inv_depth_error = (
                np.abs(rendered_disp - target_disp) * self.scene_scale
            )
            depth_row = [
                (
                    _depth_visual(target_depth, self.depth_max),
                    f"GT LiDAR Z depth (0-{self.depth_max:g})",
                ),
                *depth_row,
                (
                    _heatmap(
                        inv_depth_error,
                        _INV_DEPTH_ERROR_MAX,
                        depth_valid,
                    ),
                    f"Scaled inv-depth error (0-{_INV_DEPTH_ERROR_MAX:g})",
                ),
            ]
        depth_row.append(
            (
                _alpha_visual(
                    np.asarray(artifact.foreground_alpha, dtype=np.float32),
                    valid_mask,
                    sky_mask,
                ),
                "Foreground alpha (sky=blue, invalid=magenta)",
            )
        )
        rows.append(depth_row)

        source_height, source_width = target.shape[:2]
        tile_width = min(_TILE_MAX_WIDTH, source_width)
        tile_height = max(1, round(source_height * tile_width / source_width))
        column_count = max(len(row) for row in rows)
        panel_width = column_count * tile_width + (column_count - 1) * _GAP
        row_height = tile_height + _LABEL_HEIGHT
        panel_height = (
            _HEADER_HEIGHT
            + len(rows) * row_height
            + (len(rows) - 1) * _GAP
        )
        panel = Image.new("RGB", (panel_width, panel_height), color=(0, 0, 0))

        metric_parts = []
        for key in (
            "psnr",
            "ssim",
            "lpips",
            "bilateral_psnr",
            "ppisp_psnr",
            "depth_inv_l1",
        ):
            value = artifact.metrics.get(key)
            if value is not None and math.isfinite(float(value)):
                metric_parts.append(f"{key}={float(value):.4f}")
        title = (
            f"{artifact.split} | {artifact.image_name} | camera={artifact.camera_id} "
            f"| source={artifact.source_index} | " + "  ".join(metric_parts)
        )
        ImageDraw.Draw(panel).text(
            (10, 10), title, fill=(245, 245, 245), font=_font(17)
        )

        for row_index, row in enumerate(rows):
            y = _HEADER_HEIGHT + row_index * (row_height + _GAP)
            for column_index, (image, label) in enumerate(row):
                x = column_index * (tile_width + _GAP)
                panel.paste(_tile(image, label, tile_width, tile_height), (x, y))

        panel_path = self._panel_path(artifact)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(panel_path, format="JPEG", quality=90)

        self.records.append(
            {
                "split": artifact.split,
                "image_name": artifact.image_name,
                "image_path": artifact.image_path,
                "source_index": artifact.source_index,
                "split_image_id": artifact.split_image_id,
                "camera_id": artifact.camera_id,
                "camera_model": artifact.camera_model,
                "rig_frame_index": artifact.rig_frame_index,
                "panel_path": panel_path.relative_to(self.render_dir).as_posix(),
                "metrics": dict(artifact.metrics),
                "intrinsics": artifact.intrinsics,
                "camtoworld": artifact.camtoworld,
                "radial_coeffs": artifact.radial_coeffs,
                "tangential_coeffs": artifact.tangential_coeffs,
            }
        )

    def finalize(self, summary: Mapping[str, float]) -> Path:
        manifest_path = self.event_dir / "manifest.json"
        manifest = {
            "iteration": self.iteration,
            "completed_step": self.completed_step,
            "variant": self.variant,
            "depth_visualization_max": self.depth_max,
            "scene_scale": self.scene_scale,
            "summary": dict(summary),
            "images": self.records,
        }
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(_json_value(manifest), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return manifest_path
