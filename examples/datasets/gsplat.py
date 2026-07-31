"""Dataset adapter for the gsdata fisheye manifest format."""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from .normalize import normalize as normalize_scene


_PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def _resolve(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Dataset path escapes its root: {relative_path!r}")
    return path


def _read_pointcloud(path: Path):
    with path.open("rb") as f:
        if f.readline() != b"ply\n":
            raise ValueError(f"{path} is not a PLY file")

        vertex_count = None
        vertex_properties = []
        element = None
        binary_little_endian = False
        for _ in range(256):
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header in {path} is incomplete")
            fields = line.decode("ascii").strip().split()
            if fields[:2] == ["format", "binary_little_endian"]:
                binary_little_endian = True
            elif fields[:2] == ["element", "vertex"] and len(fields) == 3:
                element = "vertex"
                vertex_count = int(fields[2])
            elif fields[:1] == ["element"]:
                element = fields[1]
            elif fields[:1] == ["property"] and element == "vertex":
                if len(fields) != 3 or fields[1] not in _PLY_TYPES:
                    raise ValueError(f"Unsupported vertex property in {path}: {line!r}")
                vertex_properties.append((fields[2], _PLY_TYPES[fields[1]]))
            elif fields == ["end_header"]:
                break
        else:
            raise ValueError(f"PLY header in {path} is too long")

        required = {"x", "y", "z", "red", "green", "blue"}
        names = {name for name, _ in vertex_properties}
        if (
            not binary_little_endian
            or vertex_count is None
            or vertex_count <= 0
            or not required <= names
        ):
            raise ValueError(f"{path} must contain non-empty xyz/rgb vertices")

        vertices = np.fromfile(
            f, dtype=np.dtype(vertex_properties), count=vertex_count
        )
        if len(vertices) != vertex_count:
            raise ValueError(
                f"{path} declares {vertex_count} vertices but contains {len(vertices)}"
            )

    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(
        np.float32
    )
    colors = np.column_stack(
        [vertices[name] for name in ("red", "green", "blue")]
    ).astype(np.uint8)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains non-finite points")
    return points, colors


class Parser:
    """Load cameras, frames, masks, and LiDAR initialization from dataset.json."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        undistort: bool = False,
    ):
        del test_every  # Splits are explicit in the manifest.
        if factor < 1:
            raise ValueError("factor must be at least 1")
        if undistort:
            raise ValueError(
                "gsdata fisheye images must keep their calibrated distortion; "
                "pass --keep_distortion"
            )

        self.data_dir = Path(data_dir).resolve()
        manifest_path = self.data_dir / "dataset.json"
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != "gsdata-fisheye-v6":
            raise ValueError(
                f"Unsupported dataset format: {manifest.get('format')!r}"
            )

        self.factor = factor
        self.normalize = normalize
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": True}
        self.bounds = np.array([0.01, 1.0])

        self.camera_ids_by_name = {
            name: camera_id
            for camera_id, name in enumerate(manifest["cameras"])
        }
        self.camera_models_dict = {}
        self.Ks_dict = {}
        self.params_dict = {}
        self.imsize_dict = {}
        self.mask_dict = {}
        for name, camera in manifest["cameras"].items():
            if camera.get("model") != "fisheye":
                raise ValueError(f"Camera {name!r} is not a fisheye camera")
            camera_id = self.camera_ids_by_name[name]
            K = np.asarray(camera["K"], dtype=np.float32)
            radial = np.asarray(camera["radial_coeffs"], dtype=np.float32)
            width, height = int(camera["width"]), int(camera["height"])
            if K.shape != (3, 3) or radial.shape != (4,) or min(width, height) < 1:
                raise ValueError(f"Invalid calibration for camera {name!r}")
            K[:2] /= factor
            self.camera_models_dict[camera_id] = 5
            self.Ks_dict[camera_id] = K
            self.params_dict[camera_id] = radial
            self.imsize_dict[camera_id] = (width // factor, height // factor)
            self.mask_dict[camera_id] = None

        frames = []
        skipped = []
        for frame in manifest["frames"]:
            image_path = _resolve(self.data_dir, frame["image"])
            if not image_path.is_file():
                skipped.append(frame["image"])
                continue
            mask_path = _resolve(self.data_dir, frame["mask"])
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            if frame["camera"] not in self.camera_ids_by_name:
                raise ValueError(f"Unknown camera: {frame['camera']!r}")
            frames.append((frame, image_path, mask_path))
        if skipped:
            warnings.warn(
                f"Skipped {len(skipped)} missing frame(s), e.g. {skipped[:3]}",
                stacklevel=2,
            )
        if not frames:
            raise ValueError(f"No usable frames in {manifest_path}")

        self.image_names = [frame["image"] for frame, _, _ in frames]
        self.image_paths = [str(image_path) for _, image_path, _ in frames]
        self.mask_paths = [str(mask_path) for _, _, mask_path in frames]
        self.splits = [frame["split"] for frame, _, _ in frames]
        self.camera_ids = [
            self.camera_ids_by_name[frame["camera"]] for frame, _, _ in frames
        ]

        worldtocams = np.asarray(
            [frame["world_to_camera"] for frame, _, _ in frames], dtype=np.float64
        )
        if worldtocams.shape[1:] != (4, 4) or not np.isfinite(worldtocams).all():
            raise ValueError("Every world_to_camera pose must be a finite 4x4 matrix")
        self.camtoworlds = np.linalg.inv(worldtocams)

        pointcloud_path = _resolve(self.data_dir, manifest["pointcloud"])
        self.points, self.points_rgb = _read_pointcloud(pointcloud_path)
        if normalize:
            self.camtoworlds, self.points, self.transform = normalize_scene(
                self.camtoworlds, self.points
            )
        else:
            self.transform = np.eye(4)

        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        self.scene_scale = np.linalg.norm(
            camera_locations - scene_center, axis=1
        ).max()
        print(
            f"[GsplatParser] {len(frames)} images, "
            f"{len(self.camera_ids_by_name)} cameras, {len(self.points)} points."
        )


class Dataset:
    """Expose gsdata frames through the simple trainer's dataset contract."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        if load_depths:
            raise ValueError("Sparse depth supervision is not available in this dataset")
        split_name = {"train": "train", "val": "validation"}.get(split, split)
        self.parser = parser
        self.patch_size = patch_size
        self.indices = np.asarray(
            [i for i, value in enumerate(parser.splits) if value == split_name]
        )
        if not len(self.indices):
            raise ValueError(f"Dataset contains no {split_name!r} frames")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        camera_id = self.parser.camera_ids[index]
        width, height = self.parser.imsize_dict[camera_id]

        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        mask = imageio.imread(self.parser.mask_paths[index])
        if mask.ndim == 3:
            mask = mask[..., 0]
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = mask > 127
        K = self.parser.Ks_dict[camera_id].copy()

        if self.patch_size is not None:
            patch = self.patch_size
            if patch > min(height, width):
                raise ValueError(f"patch_size {patch} exceeds image size {width}x{height}")
            x = np.random.randint(0, width - patch + 1)
            y = np.random.randint(0, height - patch + 1)
            image = image[y : y + patch, x : x + patch]
            mask = mask[y : y + patch, x : x + patch]
            K[0, 2] -= x
            K[1, 2] -= y

        return {
            "camera_model": 5,
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index]).float(),
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).bool(),
            "radial_coeffs": torch.from_numpy(
                self.parser.params_dict[camera_id].copy()
            ).float(),
            "image_id": item,
        }
