import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from eval_artifacts import EvalArtifact, EvalArtifactWriter
from evaluation import masked_lpips, masked_psnr, masked_ssim
from fused_ssim import FusedSSIMMap, fused_ssim
from gaussian_models import (
    clamp_sky_sh_colors,
    composite_sky,
    create_sky_splats_with_optimizers,
    initialize_surface_priors_knn_pca,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import (
    AppearanceOptModule,
    CameraCalibrationOptModule,
    CameraOptModule,
    CameraRefinementSchedule,
    CameraRigPoseModule,
    knn,
    rgb_to_sh,
    set_random_seed,
)

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.cuda._wrapper import compute_raymap
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy, MRNFStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


_PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def _json_log_value(value):
    """Convert training metrics to strict JSON-compatible values."""
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        value = value.item() if value.numel() == 1 else value.tolist()
        return _json_log_value(value)
    if isinstance(value, np.ndarray):
        return _json_log_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_log_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_log_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_log_value(item) for item in value]
    return value


def _masked_fused_ssim_loss(
    prediction: Tensor, target: Tensor, mask: Optional[Tensor]
) -> Tensor:
    """Compute SSIM loss from windows containing only valid pixels."""
    prediction = prediction.permute(0, 3, 1, 2).contiguous()
    target = target.permute(0, 3, 1, 2).contiguous()
    if mask is None:
        return 1.0 - fused_ssim(prediction, target, padding="valid")

    ssim_map = FusedSSIMMap.apply(
        0.01**2,
        0.03**2,
        prediction,
        target,
        "valid",
        True,
    )
    valid_windows = F.avg_pool2d(
        mask.unsqueeze(1).to(dtype=ssim_map.dtype),
        kernel_size=11,
        stride=1,
    )
    valid_windows = (valid_windows >= 1.0 - 1e-6).to(dtype=ssim_map.dtype)
    valid_windows = valid_windows.expand_as(ssim_map)
    return ((1.0 - ssim_map) * valid_windows).sum() / valid_windows.sum().clamp_min(
        1.0
    )


class JsonlLogger:
    """Append rank-zero training events to a human-readable JSON Lines file."""

    def __init__(self, path: Union[str, Path], enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"

    def log(self, event: str, step: Optional[int] = None, **fields) -> None:
        if not self.enabled:
            return
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": self.run_id,
            "event": event,
        }
        if step is not None:
            record["step"] = int(step)
        record.update(
            {str(key): _json_log_value(value) for key, value in fields.items()}
        )
        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )


def _truncate_shN_for_ply(shN: Tensor, sh_degree: int) -> Tensor:
    """Keep only the higher-order SH coefficients requested for PLY export."""
    if shN.ndim != 3 or shN.shape[2] != 3:
        raise ValueError(f"shN must have shape (N, K, 3), got {tuple(shN.shape)}")
    if sh_degree < 0:
        raise ValueError("sh_degree must be non-negative")

    num_shN = (sh_degree + 1) ** 2 - 1
    if num_shN > shN.shape[1]:
        raise ValueError(
            f"Requested SH degree {sh_degree}, but shN only has "
            f"{shN.shape[1]} higher-order coefficients"
        )
    return shN[:, :num_shN, :]


def _read_lidar_ply(
    path: Path, transform: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Read the binary RGB LiDAR PLY used for Gaussian initialization."""
    if not path.is_file():
        raise FileNotFoundError(f"LiDAR initialization PLY not found: {path}")

    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            try:
                decoded = line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"Invalid PLY header: {path}") from error
            header.append(decoded)
            if decoded == "end_header":
                break

        count_line = next(
            (line for line in header if line.startswith("element vertex ")), None
        )
        if header[:1] != ["ply"] or "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Unsupported LiDAR PLY format: {path}")
        if count_line is None:
            raise ValueError(f"PLY header is missing vertex count: {path}")
        try:
            count = int(count_line.split()[2])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Invalid PLY vertex count: {path}") from error
        if count < 1:
            raise ValueError(f"LiDAR PLY is empty: {path}")

        properties = [line for line in header if line.startswith("property ")]
        expected_properties = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        if properties != expected_properties:
            raise ValueError(f"Unsupported LiDAR PLY layout: {path}")

        vertices = np.fromfile(stream, dtype=_PLY_VERTEX_DTYPE, count=count)
        if len(vertices) != count:
            raise ValueError(f"LiDAR PLY payload is truncated: {path}")
        if stream.read(1):
            raise ValueError(f"LiDAR PLY has unexpected trailing data: {path}")

    structured = vertices
    points = np.column_stack((structured["x"], structured["y"], structured["z"]))
    if not np.all(np.isfinite(points)):
        raise ValueError(f"LiDAR PLY contains non-finite coordinates: {path}")
    rgbs = np.column_stack(
        (structured["red"], structured["green"], structured["blue"])
    )

    if transform is not None:
        transform = np.asarray(transform)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("LiDAR transform must be a finite 4x4 matrix")
        points = points @ transform[:3, :3].T + transform[:3, 3]
        if not np.all(np.isfinite(points)):
            raise ValueError("LiDAR transform produced non-finite coordinates")

    return (
        np.ascontiguousarray(points, dtype=np.float32),
        np.ascontiguousarray(rgbs, dtype=np.float32) / 255.0,
    )


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Disable the held-out test/validation split while keeping test_every as the
    # interval for sampled training-set evaluation.
    use_test_split: bool = True
    # Optional inclusive frame-id range parsed from names such as *_00063-L.png.
    frame_id_min: Optional[int] = None
    frame_id_max: Optional[int] = None
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Disable dataset undistortion and train on the original (distorted) images.
    # Useful when using UT/GEER with distortion coefficients.
    keep_distortion: bool = False
    # Limit fisheye supervision to a front-facing field of view below 180 degrees.
    max_fisheye_fov: Optional[float] = None

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Maximum SH degree written to PLY. Keep this at 0 to save only the DC term
    # while retaining the full SH representation in checkpoints during training.
    ply_sh_degree: int = 0
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: Literal["sfm", "random", "lidar"] = "sfm"
    # Optional RGB LiDAR PLY. Defaults to <data_dir>/lidar.ply for init_type=lidar.
    init_lidar_path: Optional[str] = None
    # Initial number of GSs. Ignored if using sfm or lidar
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using
    # sfm or lidar
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Use LiDAR KNN-PCA to initialize surface-aligned anisotropic Gaussians.
    init_use_knn_pca: bool = False
    # Number of local points used by KNN-PCA (including the query point).
    knn_pca_k: int = 24
    # Normal-axis scale relative to the local KNN scale.
    knn_pca_normal_scale_factor: float = 0.25
    # Minimum local planarity for accepting the PCA orientation.
    knn_pca_planarity_threshold: float = 0.30
    # Maximum local curvature for accepting the PCA orientation.
    knn_pca_curvature_threshold: float = 0.10
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Train a separate Gaussian sky behind the strategy-managed foreground.
    sky_enabled: bool = False
    # PNG sky masks. Relative paths are resolved under data_dir.
    sky_mask_dir: Optional[str] = None
    # Fixed Gaussian sky geometry and trainable SH settings.
    sky_num_points: int = 100_000
    sky_radius: float = 10_000.0
    sky_initial_opacity: float = 0.7
    sky_sh0_lr: float = 2.5e-3
    sky_shN_lr: float = 2.5e-3 / 20
    # Penalize foreground opacity over conservative sky pixels.
    sky_alpha_lambda: float = 0.1

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy, MRNFStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # ---- Stability knobs ----
    # Clamp rendered colors to [0, 1] for the loss computation (helps avoid NaNs in SSIM).
    clamp_colors_for_loss: bool = True
    # Skip optimizer update if the loss is non-finite (NaN/Inf) instead of corrupting params.
    skip_non_finite_loss: bool = True
    # Clip gradient norm for splat parameters. 0 disables.
    grad_clip_norm: float = 0.0
    # Clamp the log-scales parameter (stored as log(scale)). Use wide defaults; 0 disables.
    scales_log_min: float = -12.0
    scales_log_max: float = 6.0
    # Clamp opacity logits. 0 disables.
    opacities_logit_min: float = -12.0
    opacities_logit_max: float = 12.0
    # Renormalize quaternion parameters after each optimizer step.
    renormalize_quats: bool = True

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Start camera optimization at this training step (inclusive).
    pose_opt_start_step: int = 500
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Optional independent translation / rotation learning rates. 0 falls back to pose_opt_lr.
    pose_opt_translation_lr: float = 1e-4
    pose_opt_rotation_lr: float = 1e-4
    # SO(3) is the default; 6d is retained for old experiments.
    pose_opt_rotation_mode: Literal["so3", "6d"] = "so3"
    # Reference image index in the trainset. -1 disables the single-image anchor.
    pose_opt_reference_image_id: int = 0
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Explicit physical Pose prior.
    pose_opt_prior_lambda: float = 1e-4
    pose_opt_translation_sigma: float = 0.02
    pose_opt_rotation_sigma_deg: float = 2.0
    # Replace independent image poses with one pose per rig frame.
    rig_opt: bool = False
    rig_reference_camera_id: Optional[int] = None
    rig_reference_frame_id: Optional[int] = None
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Optimize shared OPENCV_FISHEYE intrinsics and radial distortion.
    calib_opt: bool = False
    # Learning rate for focal log-scales.
    calib_opt_focal_lr: float = 1e-5
    # Learning rate for principal-point offsets normalized by focal length.
    calib_opt_principal_lr: float = 1e-5
    # Learning rate for k1, k2, k3, k4 deltas.
    calib_opt_radial_lr: float = 1e-6
    # High-order radial learning rate multiplier relative to calib_opt_radial_lr.
    calib_opt_radial_high_lr: float = 0.0
    # Regularization for all calibration deltas as optimizer weight decay.
    calib_opt_reg: float = 1e-3
    # Progressive calibration release steps.
    calib_opt_focal_start_step: int = 3_000
    calib_opt_principal_start_step: int = 8_000
    calib_opt_radial_start_step: int = 15_000
    # -1 keeps k3/k4 frozen for the complete run.
    calib_opt_high_order_start_step: int = -1
    # Freeze all camera parameters at this step; -1 disables the freeze.
    camera_freeze_step: int = 20_000
    # Shared focal scale is the default. Aspect-ratio correction is optional.
    calib_opt_shared_focal: bool = True
    calib_opt_allow_aspect_ratio: bool = False
    calib_opt_aspect_lr_scale: float = 0.1
    # Explicit calibration priors.
    calib_opt_prior_lambda: float = 1e-4
    calib_opt_focal_sigma: float = 0.03
    calib_opt_principal_sigma: float = 0.01
    calib_opt_radial_low_sigma: float = 0.01
    calib_opt_radial_high_sigma: float = 0.003
    calib_opt_aspect_sigma: float = 0.005
    # OpenCV fisheye monotonicity prior over the usable angular domain.
    calib_opt_monotonic_lambda: float = 1e-3
    calib_opt_monotonic_samples: int = 32
    calib_opt_monotonic_eps: float = 1e-3
    calib_opt_monotonic_fov_deg: float = 170.0
    # Hard bounds keep joint pose/calibration optimization identifiable.
    calib_opt_max_focal_log_scale: float = 0.1
    calib_opt_max_principal_offset: float = 0.05
    calib_opt_max_radial_delta: float = 0.1

    # Densification can be delayed until camera refinement has stabilized.
    # -1 selects an automatic camera-aware start.
    densification_start_step: int = -1

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable NVIDIA PPISP as the post-render photometric model. The first
    # integration deliberately trains only PPISP's exposure/vignetting/color/CRF
    # modules; the novel-view controller is left disabled.
    use_ppisp: bool = False

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Optional same-layout directory of LFS uint16 camera-Z depth PNGs. When
    # unset, depth_loss retains the legacy sparse COLMAP-track supervision.
    depth_dir: Optional[str] = None
    # Camera-Z value represented by uint16 65535. None reads data_dir/report.json.
    depth_max: Optional[float] = None
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Append training metrics to <result_dir>/train.log every this many steps.
    # Set to 0 to disable periodic training metric records.
    log_every: int = 100
    # Number of training steps used for loss moving averages in train.log.
    log_loss_window: int = 100

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT/3DGEER (uncented transform/planar frustum + eval 3D)
    with_ut: bool = False
    with_geer: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.pose_opt_start_step = int(self.pose_opt_start_step * factor)
        for name in (
            "calib_opt_focal_start_step",
            "calib_opt_principal_start_step",
            "calib_opt_radial_start_step",
            "calib_opt_high_order_start_step",
            "camera_freeze_step",
            "densification_start_step",
        ):
            value = getattr(self, name)
            if value >= 0:
                setattr(self, name, int(value * factor))
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MRNFStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.grow_until_iter = int(strategy.grow_until_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "simple_trainer"


def _flatten_yaml_args(
    values: Dict[str, object], prefix: Tuple[str, ...] = ()
) -> List[str]:
    args: List[str] = []
    for key, value in values.items():
        if not isinstance(key, str):
            raise SystemExit("YAML config keys must be strings.")
        path = prefix + (key.replace("_", "-"),)
        if isinstance(value, dict):
            args.extend(_flatten_yaml_args(value, path))
            continue

        if isinstance(value, bool):
            if not value:
                path = path[:-1] + (f"no-{path[-1]}",)
            args.append("--" + ".".join(path))
            continue

        flag = "--" + ".".join(path)
        args.append(flag)
        if isinstance(value, list):
            if any(isinstance(item, (dict, list)) for item in value):
                raise SystemExit(f"Unsupported nested YAML list: {flag}")
            args.extend("None" if item is None else str(item) for item in value)
        else:
            args.append("None" if value is None else str(value))
    return args


def _yaml_config_args(path: Path) -> List[str]:
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise SystemExit(f"Could not read config file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise SystemExit(f"Invalid YAML config {path}: {error}") from error
    if not isinstance(values, dict):
        raise SystemExit(f"YAML config must contain a mapping: {path}")

    preset = values.pop("preset", "default")
    if not isinstance(preset, str):
        raise SystemExit(f"YAML preset must be a string: {path}")
    return [preset, *_flatten_yaml_args(values)]


def parse_config(args: Optional[Sequence[str]] = None) -> Config:
    cli_args = list(sys.argv[1:] if args is None else args)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, overrides = parser.parse_known_args(cli_args)

    config_path = known.config
    if config_path is None and overrides and not overrides[0].startswith("-"):
        candidate = CONFIG_DIR / f"{overrides[0]}.yaml"
        if candidate.is_file():
            config_path = candidate
            overrides = overrides[1:]

    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
        "mrnf": (
            "Error-attribution densification for Eval3D and distorted cameras.",
            Config(
                with_eval3d=True,
                packed=False,
                strategy=MRNFStrategy(verbose=True),
            ),
        ),
    }
    if config_path is not None:
        base = tyro.extras.overridable_config_cli(
            configs, args=_yaml_config_args(config_path)
        )
        return tyro.cli(
            Config,
            default=base,
            args=overrides,
            config=(tyro.conf.AvoidSubcommands,),
        )
    return tyro.extras.overridable_config_cli(configs, args=overrides)


def create_splats_with_optimizers(
    parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_lidar_path: Optional[Union[str, Path]] = None,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    init_use_knn_pca: bool = False,
    knn_pca_k: int = 24,
    knn_pca_normal_scale_factor: float = 0.25,
    knn_pca_planarity_threshold: float = 0.30,
    knn_pca_curvature_threshold: float = 0.10,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_scale <= 0.0:
        raise ValueError("init_scale must be positive")
    if not 0.0 < init_opacity < 1.0:
        raise ValueError("init_opacity must be between zero and one")
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    elif init_type == "lidar":
        lidar_path = (
            Path(init_lidar_path)
            if init_lidar_path is not None
            else Path(parser.data_dir) / "lidar.ply"
        )
        lidar_points, lidar_rgbs = _read_lidar_ply(
            lidar_path, transform=getattr(parser, "transform", None)
        )
        points = torch.from_numpy(lidar_points).float()
        rgbs = torch.from_numpy(lidar_rgbs).float()
        print(f"Initializing Gaussians from LiDAR PLY: {lidar_path}")
    else:
        raise ValueError("Please specify a correct init_type: sfm, random, or lidar")

    if init_use_knn_pca:
        if init_type != "lidar":
            raise ValueError("KNN-PCA initialization currently requires init_type=lidar")
        quats, actual_scales, accepted_count = initialize_surface_priors_knn_pca(
            points,
            k=knn_pca_k,
            local_scale_factor=init_scale,
            normal_scale_factor=knn_pca_normal_scale_factor,
            planarity_threshold=knn_pca_planarity_threshold,
            curvature_threshold=knn_pca_curvature_threshold,
        )
        scales = torch.log(actual_scales)
        print(
            f"KNN-PCA surface initialization: {accepted_count}/{len(points)} "
            f"({100.0 * accepted_count / max(len(points), 1):.1f}%) points "
            f"accepted as planar; k={min(knn_pca_k, len(points))}, "
            f"normal_factor={knn_pca_normal_scale_factor}"
        )
    else:
        # Initialize the GS size to be the average dist of the 3 nearest neighbors.
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg).clamp_min(1e-6)
        scales = (
            torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
        )  # [N, 3]
        quats = torch.rand((points.shape[0], 4))  # [N, 4]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]
    quats = quats[world_rank::world_size]

    N = points.shape[0]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        if cfg.sky_enabled and cfg.sky_mask_dir is None:
            raise ValueError("sky_enabled requires sky_mask_dir")
        if cfg.sky_enabled and cfg.random_bkgd:
            raise ValueError("sky_enabled and random_bkgd cannot be used together")
        if cfg.init_use_knn_pca and cfg.init_type != "lidar":
            raise ValueError("KNN-PCA initialization currently requires init_type=lidar")
        if cfg.depth_lambda < 0.0:
            raise ValueError("depth_lambda must be non-negative")
        if cfg.depth_loss and cfg.depth_dir is not None and not cfg.keep_distortion:
            raise ValueError(
                "depth_dir sidecars require keep_distortion=True because they are "
                "aligned to the original image domain"
            )
        if cfg.sky_alpha_lambda < 0.0:
            raise ValueError("sky_alpha_lambda must be non-negative")
        if cfg.calib_opt and not cfg.keep_distortion:
            raise ValueError("calib_opt requires keep_distortion")
        if cfg.calib_opt and not cfg.with_eval3d:
            raise ValueError("calib_opt requires with_eval3d")
        if cfg.calib_opt and not (cfg.with_ut or cfg.with_geer):
            raise ValueError("calib_opt requires with_ut or with_geer")
        if cfg.calib_opt and min(
            cfg.calib_opt_max_focal_log_scale,
            cfg.calib_opt_max_principal_offset,
            cfg.calib_opt_max_radial_delta,
        ) <= 0.0:
            raise ValueError("calib_opt parameter bounds must be positive")
        if cfg.pose_opt_start_step < 0:
            raise ValueError("pose_opt_start_step must be non-negative")
        if cfg.rig_opt and not cfg.pose_opt:
            raise ValueError("rig_opt requires pose_opt")
        if cfg.pose_opt_rotation_sigma_deg <= 0.0:
            raise ValueError("pose_opt_rotation_sigma_deg must be positive")
        if cfg.pose_opt_translation_sigma <= 0.0:
            raise ValueError("pose_opt_translation_sigma must be positive")
        if cfg.pose_opt_reference_image_id < -1:
            raise ValueError("pose_opt_reference_image_id must be -1 or non-negative")
        if cfg.calib_opt_radial_high_lr < 0.0:
            raise ValueError("calib_opt_radial_high_lr must be non-negative")
        for name in (
            "calib_opt_focal_start_step",
            "calib_opt_principal_start_step",
            "calib_opt_radial_start_step",
        ):
            if getattr(cfg, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if cfg.calib_opt_high_order_start_step < -1:
            raise ValueError("calib_opt_high_order_start_step must be -1 or non-negative")
        if cfg.camera_freeze_step < -1:
            raise ValueError("camera_freeze_step must be -1 or non-negative")
        if cfg.densification_start_step < -1:
            raise ValueError("densification_start_step must be -1 or non-negative")
        appearance_methods = sum(
            (bool(cfg.app_opt), bool(cfg.use_bilateral_grid), bool(cfg.use_ppisp))
        )
        if appearance_methods > 1:
            raise ValueError(
                "app_opt, use_bilateral_grid, and use_ppisp are mutually exclusive"
            )
        if cfg.use_ppisp and cfg.batch_size != 1:
            raise ValueError("use_ppisp currently requires batch_size=1")
        if cfg.use_ppisp and cfg.patch_size is not None:
            raise ValueError(
                "use_ppisp currently requires full images (patch_size must be unset)"
            )
        if cfg.use_ppisp and world_size != 1:
            raise ValueError("use_ppisp currently supports single-GPU training only")
        if cfg.calib_opt_high_order_start_step >= 0 and (
            cfg.calib_opt_high_order_start_step < cfg.calib_opt_radial_start_step
        ):
            raise ValueError("high-order distortion cannot start before low-order distortion")
        if cfg.pose_opt_prior_lambda < 0.0 or cfg.calib_opt_prior_lambda < 0.0:
            raise ValueError("camera prior weights must be non-negative")
        if cfg.calib_opt_monotonic_lambda < 0.0:
            raise ValueError("calib_opt_monotonic_lambda must be non-negative")
        if cfg.calib_opt_monotonic_samples < 2:
            raise ValueError("calib_opt_monotonic_samples must be at least 2")
        if not 0.0 < cfg.calib_opt_monotonic_fov_deg < 180.0:
            raise ValueError("calib_opt_monotonic_fov_deg must be in (0, 180)")
        if cfg.log_every < 0:
            raise ValueError("log_every must be non-negative")
        if cfg.log_loss_window <= 0:
            raise ValueError("log_loss_window must be positive")

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Append-only structured log. Only rank zero writes in distributed runs.
        self.train_logger = JsonlLogger(
            Path(cfg.result_dir) / "train.log", enabled=world_rank == 0
        )

        # Load data: Training data should contain initial points and colors.
        from datasets.colmap import Dataset as ColmapDataset
        from datasets.colmap import Parser as ColmapParser

        parser_cls, dataset_cls = ColmapParser, ColmapDataset
        parser_kwargs = dict(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            use_test_split=cfg.use_test_split,
            undistort=not cfg.keep_distortion,
        )
        parser_kwargs["max_fisheye_fov"] = cfg.max_fisheye_fov
        parser_kwargs["frame_id_min"] = cfg.frame_id_min
        parser_kwargs["frame_id_max"] = cfg.frame_id_max
        parser_kwargs["sky_mask_dir"] = (
            cfg.sky_mask_dir if cfg.sky_enabled else None
        )
        self.parser = parser_cls(**parser_kwargs)
        self.trainset = dataset_cls(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            depth_dir=cfg.depth_dir,
            depth_max=cfg.depth_max,
        )
        self.valset = dataset_cls(
            self.parser,
            split="val",
            load_depths=cfg.depth_loss and cfg.depth_dir is not None,
            depth_dir=cfg.depth_dir,
            depth_max=cfg.depth_max,
        )
        if cfg.rig_opt:
            _, frame_counts = np.unique(self.parser.frame_ids, return_counts=True)
            if (int(frame_counts.max()) if frame_counts.size else 0) < 2:
                raise ValueError(
                    "rig_opt requires image names that group multiple cameras per frame"
                )
        if cfg.pose_opt and not cfg.rig_opt:
            if cfg.pose_opt_reference_image_id >= len(self.trainset):
                raise ValueError("pose_opt_reference_image_id is outside the trainset")
        # Render a lightweight, deterministic sample of training images at eval
        # time. When the held-out split is disabled, trainset contains all images.
        self.train_evalset = torch.utils.data.Subset(
            self.trainset,
            range(0, len(self.trainset), cfg.test_every),
        )
        if world_rank == 0:
            print(
                f"[Eval] {len(self.valset)} validation images; "
                f"{len(self.train_evalset)} sampled training images "
                f"(every {cfg.test_every})."
            )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.camera_schedule = CameraRefinementSchedule(
            pose_start=cfg.pose_opt_start_step,
            focal_start=cfg.calib_opt_focal_start_step,
            principal_start=cfg.calib_opt_principal_start_step,
            radial_start=cfg.calib_opt_radial_start_step,
            high_order_start=cfg.calib_opt_high_order_start_step,
            freeze_step=cfg.camera_freeze_step,
        )
        if cfg.densification_start_step >= 0:
            self.densification_start_step = cfg.densification_start_step
        elif cfg.calib_opt:
        # elif cfg.pose_opt or cfg.calib_opt:
            self.densification_start_step = max(
                cfg.pose_opt_start_step + 1000,
                cfg.calib_opt_focal_start_step,
            )
        else:
            self.densification_start_step = 0

        # Precompute a fisheye valid-pixel mask (raymap.valid_flag) when possible.
        # Pixels outside the valid domain yield zero rays from `compute_raymap()`,
        # which can destabilize training if included in the loss.
        self._fisheye_valid_mask_cache: Dict[Tuple, Tensor] = {}

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_lidar_path=cfg.init_lidar_path,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            init_use_knn_pca=cfg.init_use_knn_pca,
            knn_pca_k=cfg.knn_pca_k,
            knn_pca_normal_scale_factor=cfg.knn_pca_normal_scale_factor,
            knn_pca_planarity_threshold=cfg.knn_pca_planarity_threshold,
            knn_pca_curvature_threshold=cfg.knn_pca_curvature_threshold,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.sky_splats = None
        self.sky_optimizers: Dict[str, torch.optim.Optimizer] = {}
        if cfg.sky_enabled:
            self.sky_splats, self.sky_optimizers = create_sky_splats_with_optimizers(
                count=cfg.sky_num_points,
                radius=cfg.sky_radius,
                initial_opacity=cfg.sky_initial_opacity,
                sh_degree=cfg.sh_degree,
                sh0_lr=cfg.sky_sh0_lr,
                shN_lr=cfg.sky_shN_lr,
                seed=42,
                device=self.device,
                world_rank=world_rank,
                world_size=world_size,
            )
            print(
                "Sky model initialized. Number of sky GS:",
                len(self.sky_splats["means"]),
            )

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, (DefaultStrategy, MRNFStrategy)):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        if isinstance(self.cfg.strategy, MRNFStrategy):
            if not cfg.with_eval3d:
                raise ValueError("MRNFStrategy requires with_eval3d=True")
            if cfg.packed:
                raise ValueError("MRNFStrategy does not support packed rasterization")
            if world_size > 1:
                raise ValueError("MRNFStrategy does not support distributed rasterization")

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            reference_image_id = (
                None if cfg.pose_opt_reference_image_id < 0 else cfg.pose_opt_reference_image_id
            )
            if cfg.rig_opt:
                self.pose_adjust = CameraRigPoseModule(
                    torch.from_numpy(self.parser.camtoworlds),
                    torch.from_numpy(self.parser.frame_ids),
                    torch.tensor(self.parser.camera_ids),
                    reference_camera_id=cfg.rig_reference_camera_id,
                    reference_frame_id=cfg.rig_reference_frame_id,
                    rotation_mode=cfg.pose_opt_rotation_mode,
                ).to(self.device)
            else:
                self.pose_adjust = CameraOptModule(
                    len(self.trainset),
                    reference_index=reference_image_id,
                    rotation_mode=cfg.pose_opt_rotation_mode,
                ).to(self.device)
            self.pose_adjust.zero_init()
            translation_lr = (
                cfg.pose_opt_translation_lr
                if cfg.pose_opt_translation_lr > 0.0
                else cfg.pose_opt_lr
            )
            rotation_lr = (
                cfg.pose_opt_rotation_lr
                if cfg.pose_opt_rotation_lr > 0.0
                else cfg.pose_opt_lr
            )
            self.pose_optimizers = [
                torch.optim.Adam(
                    [
                        {
                            "params": self.pose_adjust.trans.parameters(),
                            "lr": translation_lr * math.sqrt(cfg.batch_size),
                        },
                        {
                            "params": self.pose_adjust.rot.parameters(),
                            "lr": rotation_lr * math.sqrt(cfg.batch_size),
                        },
                    ],
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        self.calibration_optimizers = []
        if cfg.calib_opt:
            camera_ids = sorted(set(self.parser.camera_ids))
            unsupported_camera_ids = [
                camera_id
                for camera_id in camera_ids
                if self.parser.camera_models_dict[camera_id] != 5
            ]
            if unsupported_camera_ids:
                raise ValueError(
                    "calib_opt only supports OPENCV_FISHEYE cameras; "
                    f"unsupported camera ids: {unsupported_camera_ids}"
                )
            self.calibration_camera_indices = {
                camera_id: index for index, camera_id in enumerate(camera_ids)
            }
            self.calibration_adjust = CameraCalibrationOptModule(
                len(camera_ids),
                shared_focal=cfg.calib_opt_shared_focal,
                allow_aspect_ratio=cfg.calib_opt_allow_aspect_ratio,
            ).to(self.device)
            self.calibration_optimizers = [
                torch.optim.Adam(
                    [
                        {
                            "params": (
                                self.calibration_adjust.focal_log_scales.parameters()
                            ),
                            "lr": cfg.calib_opt_focal_lr * math.sqrt(cfg.batch_size),
                        },
                        {
                            "params": (
                                self.calibration_adjust.principal_offsets.parameters()
                            ),
                            "lr": cfg.calib_opt_principal_lr
                            * math.sqrt(cfg.batch_size),
                        },
                        {
                            "params": (
                                self.calibration_adjust.radial_deltas.parameters()
                            ),
                            "lr": cfg.calib_opt_radial_lr * math.sqrt(cfg.batch_size),
                        },
                    ],
                    weight_decay=cfg.calib_opt_reg,
                )
            ]
            if world_size > 1:
                self.calibration_adjust = DDP(self.calibration_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(
                len(self.trainset),
                reference_index=None,
                rotation_mode=cfg.pose_opt_rotation_mode,
            ).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        self.ppisp = None
        self.ppisp_optimizers = []
        self.ppisp_camera_ids: Tuple[int, ...] = ()
        self.ppisp_camera_id_to_index: Dict[int, int] = {}
        self.ppisp_frame_indices: Optional[Tensor] = None
        self.ppisp_frames_per_camera: List[int] = []
        self.ppisp_version: Optional[str] = None
        if cfg.use_ppisp:
            try:
                import ppisp as ppisp_package
                from ppisp import PPISP, PPISPConfig
            except ImportError as error:
                raise RuntimeError(
                    "use_ppisp requires the PPISP package in the active Python environment"
                ) from error

            self.ppisp_version = ppisp_package.__version__
            self.ppisp_camera_ids = tuple(
                sorted({int(camera_id) for camera_id in self.parser.camera_ids})
            )
            self.ppisp_camera_id_to_index = {
                camera_id: index
                for index, camera_id in enumerate(self.ppisp_camera_ids)
            }

            # PPISP's per-frame parameters belong to individual camera images,
            # not to the rig timestamp shared by the left/right cameras. Group
            # the indices by camera so the official per-camera reports can slice
            # the parameter arrays correctly.
            train_camera_ids = [
                int(self.parser.camera_ids[int(dataset_index)])
                for dataset_index in self.trainset.indices
            ]
            frame_indices = np.empty(len(self.trainset), dtype=np.int64)
            next_frame_index = 0
            for camera_id in self.ppisp_camera_ids:
                image_ids_for_camera = [
                    image_id
                    for image_id, train_camera_id in enumerate(train_camera_ids)
                    if train_camera_id == camera_id
                ]
                self.ppisp_frames_per_camera.append(len(image_ids_for_camera))
                for image_id in image_ids_for_camera:
                    frame_indices[image_id] = next_frame_index
                    next_frame_index += 1
            if next_frame_index != len(self.trainset):
                raise RuntimeError("Failed to index all PPISP training frames")
            self.ppisp_frame_indices = torch.from_numpy(frame_indices).to(
                device=self.device
            )

            ppisp_config = PPISPConfig(
                use_controller=False,
                scheduler_decay_max_steps=cfg.max_steps,
            )
            self.ppisp = PPISP(
                num_cameras=len(self.ppisp_camera_ids),
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)
            self.ppisp_optimizers = self.ppisp.create_optimizers()

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def _apply_calibration_adjustment(
        self, Ks: Tensor, radial_coeffs: Tensor, camera_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Map raw COLMAP camera ids to shared learnable calibration rows."""
        calibration_ids = torch.tensor(
            [
                self.calibration_camera_indices[int(camera_id)]
                for camera_id in camera_ids.detach().cpu().reshape(-1).tolist()
            ],
            dtype=torch.long,
            device=Ks.device,
        ).reshape(camera_ids.shape)
        return self.calibration_adjust(Ks, radial_coeffs, calibration_ids)

    def _calibration_module(self) -> CameraCalibrationOptModule:
        if self.world_size > 1:
            return self.calibration_adjust.module
        return self.calibration_adjust

    def _pose_module(self):
        if self.world_size > 1:
            return self.pose_adjust.module
        return self.pose_adjust

    def _apply_bilateral_grid(self, colors: Tensor, image_ids: Tensor) -> Tensor:
        """Apply the learned per-image spatial color correction."""
        height, width = colors.shape[1:3]
        grid_y, grid_x = torch.meshgrid(
            (torch.arange(height, device=colors.device) + 0.5) / height,
            (torch.arange(width, device=colors.device) + 0.5) / width,
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        return slice(
            self.bil_grids,
            grid_xy.expand(colors.shape[0], -1, -1, -1),
            colors,
            image_ids.unsqueeze(-1),
        )["rgb"]

    def _apply_ppisp(
        self,
        colors: Tensor,
        camera_ids: Tensor,
        image_ids: Optional[Tensor],
    ) -> Tensor:
        """Apply PPISP using a training-frame or novel-view photometric state."""
        if self.ppisp is None or self.ppisp_frame_indices is None:
            raise RuntimeError("PPISP is not initialized")
        if colors.shape[0] != 1 or camera_ids.numel() != 1:
            raise ValueError("PPISP currently expects one full image per batch")

        raw_camera_id = int(camera_ids.reshape(-1)[0].item())
        try:
            camera_index = self.ppisp_camera_id_to_index[raw_camera_id]
        except KeyError as error:
            raise ValueError(f"Unknown PPISP camera id: {raw_camera_id}") from error

        frame_index = -1
        if image_ids is not None:
            if image_ids.numel() != 1:
                raise ValueError("PPISP currently expects one image id per batch")
            image_id = int(image_ids.reshape(-1)[0].item())
            if not 0 <= image_id < len(self.ppisp_frame_indices):
                raise ValueError(f"Unknown PPISP training image id: {image_id}")
            frame_index = int(self.ppisp_frame_indices[image_id].item())

        return self.ppisp(
            colors.squeeze(0),
            camera_idx=camera_index,
            frame_idx=frame_index,
        ).unsqueeze(0)

    def _camera_stage(self, step: int):
        return self.camera_schedule.at(step, self.cfg.pose_opt, self.cfg.calib_opt)

    def _apply_pose_adjustment(
        self,
        camtoworlds: Tensor,
        image_ids: Tensor,
        frame_ids: Optional[Tensor] = None,
        camera_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if self.cfg.rig_opt:
            if frame_ids is None or camera_ids is None:
                raise RuntimeError("rig_opt requires frame_id and camera_id tensors")
            return self.pose_adjust(camtoworlds, frame_ids, camera_ids)
        return self.pose_adjust(camtoworlds, image_ids)

    def _camera_regularization(
        self,
        stage,
        adjusted_radial_coeffs: Optional[Tensor],
    ) -> Tensor:
        loss = torch.zeros((), device=self.device)
        cfg = self.cfg
        if cfg.pose_opt and stage.pose:
            loss = loss + cfg.pose_opt_prior_lambda * self._pose_module().prior_loss(
                cfg.pose_opt_translation_sigma,
                math.radians(cfg.pose_opt_rotation_sigma_deg),
            )
        if cfg.calib_opt and not stage.frozen and (
            stage.focal or stage.principal or stage.radial_low or stage.radial_high
        ):
            calibration = self._calibration_module()
            loss = loss + cfg.calib_opt_prior_lambda * calibration.prior_loss(
                cfg.calib_opt_focal_sigma,
                cfg.calib_opt_principal_sigma,
                cfg.calib_opt_radial_low_sigma,
                cfg.calib_opt_radial_high_sigma,
                cfg.calib_opt_aspect_sigma,
            )
            if (
                adjusted_radial_coeffs is not None
                and (stage.radial_low or stage.radial_high)
                and cfg.calib_opt_monotonic_lambda > 0.0
            ):
                theta_max = math.radians(cfg.calib_opt_monotonic_fov_deg / 2.0)
                loss = loss + cfg.calib_opt_monotonic_lambda * calibration.monotonicity_loss(
                    adjusted_radial_coeffs,
                    theta_max=theta_max,
                    samples=cfg.calib_opt_monotonic_samples,
                    eps=cfg.calib_opt_monotonic_eps,
                )
        return loss

    def _apply_camera_gradient_controls(self, stage):
        if not self.cfg.calib_opt:
            return
        calibration = self._calibration_module()
        radial_lr = max(self.cfg.calib_opt_radial_lr, 1e-12)
        calibration.apply_gradient_controls(
            focal_active=stage.focal,
            principal_active=stage.principal,
            radial_low_active=stage.radial_low,
            radial_high_active=stage.radial_high,
            radial_high_lr_scale=self.cfg.calib_opt_radial_high_lr / radial_lr,
            aspect_lr_scale=self.cfg.calib_opt_aspect_lr_scale,
        )

    def _project_camera_parameters(self):
        if self.cfg.calib_opt:
            self._calibration_module().project_parameters()

    @torch.no_grad()
    def _camera_metrics(self) -> Dict[str, Tensor]:
        metrics = {}
        if self.cfg.pose_opt:
            metrics.update({f"pose/{k}": v for k, v in self._pose_module().metrics().items()})
        if self.cfg.calib_opt:
            base_focal = torch.tensor(
                [
                    [
                        self.parser.Ks_dict[camera_id][0, 0],
                        self.parser.Ks_dict[camera_id][1, 1],
                    ]
                    for camera_id in sorted(set(self.parser.camera_ids))
                ],
                device=self.device,
                dtype=torch.float32,
            )
            metrics.update(
                {
                    f"calib/{k}": v
                    for k, v in self._calibration_module().metrics(base_focal).items()
                }
            )
        return metrics

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        splats: Optional[torch.nn.ParameterDict] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        is_foreground = splats is None
        calc_densification_info = bool(
            kwargs.pop("calc_densification_info", False)
        ) and is_foreground
        splats = self.splats if splats is None else splats
        means = splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if is_foreground and self.cfg.app_opt:
            colors = self.app_module(
                features=splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = "pinhole"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if is_foreground and isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad if is_foreground else False,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            with_ut=self.cfg.with_ut,
            with_geer=self.cfg.with_geer,
            with_eval3d=self.cfg.with_eval3d,
            calc_densification_info=calc_densification_info,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def render_scene(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Render foreground and, when enabled, composite an independent sky."""
        backgrounds = kwargs.pop("backgrounds", None)
        foreground_kwargs = dict(kwargs)
        if self.sky_splats is None and backgrounds is not None:
            foreground_kwargs["backgrounds"] = backgrounds
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            masks=masks,
            rasterize_mode=rasterize_mode,
            camera_model=camera_model,
            **foreground_kwargs,
        )
        if self.sky_splats is None:
            return renders, alphas, info

        sky_kwargs = dict(kwargs)
        sky_kwargs["render_mode"] = "RGB"
        if backgrounds is not None:
            sky_kwargs["backgrounds"] = backgrounds
        sky_colors, _, _ = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            masks=masks,
            rasterize_mode=rasterize_mode,
            camera_model=camera_model,
            splats=self.sky_splats,
            **sky_kwargs,
        )
        foreground_colors = renders[..., :3]
        colors = composite_sky(foreground_colors, alphas, sky_colors[..., :3])
        if renders.shape[-1] > 3:
            renders = torch.cat((colors, renders[..., 3:]), dim=-1)
        else:
            renders = colors
        return renders, alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                if isinstance(cfg.strategy, MRNFStrategy):
                    preset = "mrnf"
                elif isinstance(cfg.strategy, MCMCStrategy):
                    preset = "mcmc"
                else:
                    preset = "default"
                values = asdict(cfg)
                values["steps_scaler"] = 1.0
                yaml.safe_dump({"preset": preset, **values}, f, sort_keys=False)

        max_steps = cfg.max_steps
        init_step = 0
        self.train_logger.log(
            "run_start",
            step=init_step,
            max_steps=max_steps,
            world_size=world_size,
            config_path=Path(cfg.result_dir) / "cfg.yml",
            ppisp_version=self.ppisp_version,
            ppisp_camera_ids=self.ppisp_camera_ids,
            ppisp_frames_per_camera=self.ppisp_frames_per_camera,
        )

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        pose_scheduler = None
        if cfg.pose_opt and cfg.pose_opt_start_step < max_steps:
            # pose optimization has a learning rate schedule
            pose_opt_steps = max_steps - cfg.pose_opt_start_step
            pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.pose_optimizers[0], gamma=0.01 ** (1.0 / pose_opt_steps)
            )
        calibration_scheduler = None
        if cfg.calib_opt:
            calibration_steps = max(
                max_steps - cfg.calib_opt_focal_start_step,
                1,
            )
            calibration_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.calibration_optimizers[0],
                gamma=0.01 ** (1.0 / calibration_steps),
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )
        ppisp_schedulers = []
        if self.ppisp is not None:
            ppisp_schedulers = self.ppisp.create_schedulers(
                self.ppisp_optimizers,
                max_steps,
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        loss_history = deque(maxlen=cfg.log_loss_window)
        l1loss_history = deque(maxlen=cfg.log_loss_window)
        ssimloss_history = deque(maxlen=cfg.log_loss_window)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            camera_ids = data["camera_id"].to(device)
            frame_ids = data["frame_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            sky_masks = (
                data["sky_mask"].to(device) if "sky_mask" in data else None
            )  # [1, H, W]
            if cfg.sky_enabled and sky_masks is None:
                raise RuntimeError("Gaussian sky is enabled but the batch has no sky_mask")
            if masks is not None:
                pixels = pixels * masks.unsqueeze(-1)
            radial_coeffs = (
                data["radial_coeffs"].to(device)
                if (cfg.keep_distortion and "radial_coeffs" in data)
                else None
            )
            tangential_coeffs = (
                data["tangential_coeffs"].to(device)
                if (cfg.keep_distortion and "tangential_coeffs" in data)
                else None
            )
            if cfg.depth_loss:
                if "depth" in data:
                    depth_map_gt = data["depth"].to(device)  # [1, H, W]
                    points = depths_gt = None
                else:
                    depth_map_gt = None
                    points = data["points"].to(device)  # [1, M, 2]
                    depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]
            stage = self._camera_stage(step)

            valid_f = None
            if data["camera_model"] == 5 and radial_coeffs is not None:
                # Keep this mask tied to the raw COLMAP model. Camera corrections
                # are deliberately bounded and the raw mask is a stable, cheap
                # conservative support for all correction stages.
                K_key = tuple(float(x) for x in data["K"].flatten().tolist())
                radial_key = tuple(float(x) for x in data["radial_coeffs"].flatten().tolist())
                key = ("fisheye_valid_mask", int(width), int(height), K_key, radial_key)
                valid_mask = self._fisheye_valid_mask_cache.get(key, None)
                if valid_mask is None:
                    with torch.no_grad():
                        raymap = compute_raymap(
                            Ks=Ks,
                            width=int(width),
                            height=int(height),
                            camera_model="fisheye",
                            radial_coeffs=radial_coeffs,
                        )  # [1, H, W, 3]
                        valid_mask = torch.linalg.vector_norm(raymap, dim=-1) > 1e-6
                    self._fisheye_valid_mask_cache[key] = valid_mask
                masks = valid_mask if masks is None else masks & valid_mask
                if sky_masks is not None:
                    sky_masks = sky_masks & valid_mask
                # Zero-out invalid pixels for both prediction and target.
                valid_f = valid_mask.unsqueeze(-1).to(dtype=pixels.dtype)  # [1,H,W,1]
                pixels = pixels * valid_f

            if cfg.calib_opt:
                if radial_coeffs is None:
                    raise RuntimeError(
                        "calib_opt requires a four-coefficient fisheye calibration"
                    )
                Ks, radial_coeffs = self._apply_calibration_adjustment(
                    Ks, radial_coeffs, camera_ids
                )

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            pose_opt_active = stage.pose
            if cfg.pose_opt:
                camtoworlds = self._apply_pose_adjustment(
                    camtoworlds, image_ids, frame_ids, camera_ids
                )

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            densification_active = step >= self.densification_start_step
            collect_mrnf_info = (
                isinstance(cfg.strategy, MRNFStrategy)
                and densification_active
                and cfg.strategy.should_collect(step)
            )

            # forward
            renders, alphas, info = self.render_scene(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
                camera_model="fisheye" if cfg.keep_distortion and data["camera_model"] == 5 else "pinhole",
                radial_coeffs=radial_coeffs,
                tangential_coeffs=tangential_coeffs,
                calc_densification_info=collect_mrnf_info,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if valid_f is not None:
                colors = colors * valid_f

            if cfg.use_bilateral_grid:
                colors = self._apply_bilateral_grid(colors, image_ids)

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if cfg.use_ppisp:
                colors = self._apply_ppisp(colors, camera_ids, image_ids)

            # loss
            colors_for_loss = (
                colors.clamp(0.0, 1.0) if cfg.clamp_colors_for_loss else colors
            )
            if densification_active:
                if isinstance(cfg.strategy, MRNFStrategy):
                    cfg.strategy.step_pre_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        rendered=colors_for_loss,
                        target=pixels,
                        mask=masks,
                    )
                else:
                    cfg.strategy.step_pre_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                    )
            if masks is None:
                l1loss = F.l1_loss(colors_for_loss, pixels)
            else:
                valid = masks.unsqueeze(-1)
                l1loss = (
                    (colors_for_loss - pixels).abs().masked_select(valid).mean()
                )
            if cfg.ssim_lambda:
                ssimloss = _masked_fused_ssim_loss(
                    colors_for_loss, pixels, masks
                )
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            else:
                ssimloss = torch.zeros_like(l1loss)
                loss = l1loss
            ppisp_reg_loss = torch.zeros_like(l1loss)
            if self.ppisp is not None:
                ppisp_reg_loss = self.ppisp.get_regularization_loss()
                loss += ppisp_reg_loss
            if cfg.depth_loss:
                if depth_map_gt is not None:
                    rendered_depth = depths.squeeze(-1)
                    depth_valid = depth_map_gt > 0.0
                    if masks is not None:
                        depth_valid &= masks
                    rendered_disp = torch.where(
                        rendered_depth > 0.0,
                        rendered_depth.clamp_min(1e-8).reciprocal(),
                        torch.zeros_like(rendered_depth),
                    )
                    target_disp = torch.where(
                        depth_valid,
                        depth_map_gt.clamp_min(1e-8).reciprocal(),
                        torch.zeros_like(depth_map_gt),
                    )
                    depth_weight = depth_valid.to(rendered_depth.dtype)
                    depthloss = (
                        ((rendered_disp - target_disp).abs() * depth_weight).sum()
                        / depth_weight.sum().clamp_min(1.0)
                        * self.scene_scale
                    )
                else:
                    assert points is not None and depths_gt is not None
                    # Query rendered depths at projected COLMAP tracks.
                    points = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    )  # normalize to [-1, 1]
                    grid = points.unsqueeze(2)  # [1, M, 1, 2]
                    sampled_depths = F.grid_sample(
                        depths.permute(0, 3, 1, 2), grid, align_corners=True
                    )  # [1, 1, M, 1]
                    sampled_depths = sampled_depths.squeeze(3).squeeze(1)  # [1, M]
                    # Calculate loss in disparity space.
                    disp = torch.where(
                        sampled_depths > 0.0,
                        sampled_depths.clamp_min(1e-8).reciprocal(),
                        torch.zeros_like(sampled_depths),
                    )
                    disp_gt = depths_gt.clamp_min(1e-8).reciprocal()  # [1, M]
                    depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()
            sky_alpha_loss = torch.zeros_like(l1loss)
            if (
                self.sky_splats is not None
                and sky_masks is not None
                and cfg.sky_alpha_lambda > 0.0
            ):
                sky_valid = sky_masks.unsqueeze(-1).to(dtype=alphas.dtype)
                sky_alpha_loss = (alphas * sky_valid).sum() / sky_valid.sum().clamp_min(
                    1.0
                )
                loss += cfg.sky_alpha_lambda * sky_alpha_loss
            loss += self._camera_regularization(stage, radial_coeffs)

            do_update = True
            if cfg.skip_non_finite_loss and (not torch.isfinite(loss).all()):
                do_update = False
                if world_rank == 0:
                    print(
                        f"Step {step}: non-finite loss detected (loss={loss.item()}). "
                        "Skipping optimizer update."
                    )
                    self.train_logger.log(
                        "non_finite_loss",
                        step=step,
                        loss=loss,
                        update_applied=False,
                    )
            if do_update:
                loss.backward()

                self._apply_camera_gradient_controls(stage)

                # Some rendering modes (e.g. UT / eval3d) do not provide a differentiable
                # `info["means2d"]` tensor for gradient-based refinement strategies.
                # Cache the current means gradients before they are cleared by `zero_grad()`
                # so strategies can still build a proxy signal.
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    means_grad = self.splats["means"].grad
                    if means_grad is not None:
                        info["_strategy_means_grad"] = means_grad.detach()

                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0.0:
                    parameters = list(self.splats.parameters())
                    if self.sky_splats is not None:
                        parameters.extend(
                            parameter
                            for parameter in self.sky_splats.parameters()
                            if parameter.requires_grad
                        )
                    torch.nn.utils.clip_grad_norm_(
                        parameters, max_norm=cfg.grad_clip_norm
                    )

            loss_value = loss.detach().item()
            l1loss_value = l1loss.detach().item()
            ssimloss_value = ssimloss.detach().item()
            if math.isfinite(loss_value):
                loss_history.append(loss_value)
            if math.isfinite(l1loss_value):
                l1loss_history.append(l1loss_value)
            if math.isfinite(ssimloss_value):
                ssimloss_history.append(ssimloss_value)

            desc = (
                f"loss={loss.item():.3f}| stage={stage.name}| "
                f"sh degree={sh_degree_to_use}| "
            )
            if self.sky_splats is not None:
                desc += f"sky alpha={sky_alpha_loss.item():.4f}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if self.ppisp is not None:
                desc += f"ppisp reg={ppisp_reg_loss.item():.4f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.log_every > 0 and step % cfg.log_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                image_ids_cpu = image_ids.detach().cpu().reshape(-1).tolist()
                image_details = []
                for image_id in image_ids_cpu:
                    dataset_index = int(self.trainset.indices[int(image_id)])
                    image_details.append(
                        {
                            "image_id": int(image_id),
                            "image_name": self.parser.image_names[dataset_index],
                            "image_path": self.parser.image_paths[dataset_index],
                        }
                    )
                metrics = {
                    "loss": loss_value,
                    "l1_loss": l1loss_value,
                    "ssim_loss": ssimloss_value,
                    "loss_ma": (
                        sum(loss_history) / len(loss_history)
                        if loss_history
                        else None
                    ),
                    "l1_loss_ma": (
                        sum(l1loss_history) / len(l1loss_history)
                        if l1loss_history
                        else None
                    ),
                    "ssim_loss_ma": (
                        sum(ssimloss_history) / len(ssimloss_history)
                        if ssimloss_history
                        else None
                    ),
                    "images": image_details,
                    "num_gaussians": len(self.splats["means"]),
                    "gpu_memory_gib": mem,
                    "camera_stage": stage.name,
                    "sh_degree": sh_degree_to_use,
                    "means_lr": schedulers[0].get_last_lr()[0],
                    "update_applied": do_update,
                }
                for name, value in self._camera_metrics().items():
                    metrics[name] = value
                if cfg.calib_opt:
                    calibration = self._calibration_module()
                    metrics["calib_focal_log_scale_max"] = (
                        calibration.focal_log_scales.weight.detach().abs().max()
                    )
                    metrics["calib_principal_offset_max"] = (
                        calibration.principal_offsets.weight.detach().abs().max()
                    )
                    metrics["calib_radial_delta_max"] = (
                        calibration.radial_deltas.weight.detach().abs().max()
                    )
                if self.sky_splats is not None:
                    metrics["sky_alpha_loss"] = (
                        cfg.sky_alpha_lambda * sky_alpha_loss.item()
                    )
                    metrics["foreground_alpha_on_sky"] = sky_alpha_loss.item()
                    metrics["num_sky_gaussians"] = len(self.sky_splats["means"])
                if cfg.depth_loss:
                    metrics["depth_loss"] = depthloss.item()
                if cfg.use_bilateral_grid:
                    metrics["tv_loss"] = tvloss.item()
                if self.ppisp is not None:
                    metrics["ppisp_regularization_loss"] = ppisp_reg_loss.item()
                    metrics["ppisp_lr"] = ppisp_schedulers[0].get_last_lr()[0]
                    metrics["ppisp_exposure_mean"] = (
                        self.ppisp.exposure_params.detach().mean()
                    )
                    metrics["ppisp_exposure_abs_max"] = (
                        self.ppisp.exposure_params.detach().abs().max()
                    )
                    metrics["ppisp_color_abs_max"] = (
                        self.ppisp.color_params.detach().abs().max()
                    )
                    metrics["ppisp_vignetting_alpha_abs_max"] = (
                        self.ppisp.vignetting_params[..., 2:].detach().abs().max()
                    )
                self.train_logger.log("train", step=step, **metrics)

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                if self.sky_splats is not None:
                    stats["num_sky_GS"] = len(self.sky_splats["means"])
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if self.sky_splats is not None:
                    data["sky_splats"] = self.sky_splats.state_dict()
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.calib_opt:
                    data["calibration_adjust"] = (
                        self._calibration_module().state_dict()
                    )
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                if cfg.use_bilateral_grid:
                    data["bilateral_grid"] = self.bil_grids.state_dict()
                if self.ppisp is not None:
                    data["ppisp"] = self.ppisp.state_dict()
                    data["ppisp_camera_ids"] = self.ppisp_camera_ids
                    data["ppisp_frame_indices"] = (
                        self.ppisp_frame_indices.detach().cpu()
                    )
                checkpoint_path = (
                    f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
                torch.save(data, checkpoint_path)
                self.train_logger.log(
                    "checkpoint",
                    step=step,
                    path=checkpoint_path,
                    **stats,
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if not 0 <= cfg.ply_sh_degree <= cfg.sh_degree:
                    raise ValueError(
                        "ply_sh_degree must be between 0 and sh_degree"
                    )

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    # Post-render appearance adapters (bilateral grid or PPISP)
                    # are not representable in PLY. Keep the exported Gaussian
                    # colors on the canonical pre-adapter SH gauge.
                    sh0 = self.splats["sh0"]
                    shN = _truncate_shN_for_ply(
                        self.splats["shN"], cfg.ply_sh_degree
                    )

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )
                if self.sky_splats is not None:
                    export_splats(
                        means=self.sky_splats["means"],
                        scales=self.sky_splats["scales"],
                        quats=self.sky_splats["quats"],
                        opacities=self.sky_splats["opacities"],
                        sh0=self.sky_splats["sh0"],
                        shN=_truncate_shN_for_ply(
                            self.sky_splats["shN"], cfg.ply_sh_degree
                        ),
                        format="ply",
                        save_to=f"{self.ply_dir}/sky_{step}.ply",
                    )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            if do_update:
                for optimizer in self.optimizers.values():
                    if cfg.visible_adam:
                        optimizer.step(visibility_mask)
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            for optimizer in self.sky_optimizers.values():
                if do_update:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                if do_update and pose_opt_active:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.calibration_optimizers:
                calibration_opt_active = (
                    stage.focal or stage.principal or stage.radial_low or stage.radial_high
                )
                if do_update and calibration_opt_active and not stage.frozen:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                if do_update:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                if do_update:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.ppisp_optimizers:
                if do_update:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if do_update:
                for scheduler in schedulers:
                    scheduler.step()
                for scheduler in ppisp_schedulers:
                    scheduler.step()
                if pose_opt_active and pose_scheduler is not None:
                    pose_scheduler.step()
                if (
                    cfg.calib_opt
                    and calibration_scheduler is not None
                    and (stage.focal or stage.principal or stage.radial_low or stage.radial_high)
                ):
                    calibration_scheduler.step()

                # Post-step parameter clamps for numerical stability.
                if cfg.scales_log_max > cfg.scales_log_min:
                    self.splats["scales"].data.clamp_(
                        min=cfg.scales_log_min, max=cfg.scales_log_max
                    )
                if cfg.opacities_logit_max > cfg.opacities_logit_min:
                    self.splats["opacities"].data.clamp_(
                        min=cfg.opacities_logit_min, max=cfg.opacities_logit_max
                    )
                if cfg.renormalize_quats:
                    self.splats["quats"].data = F.normalize(
                        self.splats["quats"].data, dim=-1, eps=1e-12
                    )
                if self.sky_splats is not None:
                    clamp_sky_sh_colors(self.sky_splats, cfg.sh_degree)
                if cfg.calib_opt:
                    calibration = self._calibration_module()
                    calibration.focal_log_scales.weight.data.clamp_(
                        -cfg.calib_opt_max_focal_log_scale,
                        cfg.calib_opt_max_focal_log_scale,
                    )
                    calibration.principal_offsets.weight.data.clamp_(
                        -cfg.calib_opt_max_principal_offset,
                        cfg.calib_opt_max_principal_offset,
                    )
                    calibration.radial_deltas.weight.data.clamp_(
                        -cfg.calib_opt_max_radial_delta,
                        cfg.calib_opt_max_radial_delta,
                    )
                    calibration.project_parameters()
                if cfg.pose_opt:
                    self._pose_module()._zero_reference()

            # Run post-backward steps after backward and optimizer
            if do_update and densification_active:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                    )
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        lr=schedulers[0].get_last_lr()[0],
                    )
                elif isinstance(self.cfg.strategy, MRNFStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                    )
                else:
                    assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

        self.train_logger.log(
            "run_complete",
            step=max_steps - 1,
            elapsed_seconds=time.time() - global_tic,
            num_gaussians=len(self.splats["means"]),
            num_sky_gaussians=(
                len(self.sky_splats["means"]) if self.sky_splats is not None else 0
            ),
        )

    def _append_image_metrics(
        self,
        metrics,
        prefix: str,
        prediction: Tensor,
        target: Tensor,
        mask: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        values = {
            f"{prefix}psnr": masked_psnr(prediction, target, mask),
            f"{prefix}ssim": masked_ssim(self.ssim, prediction, target, mask),
            f"{prefix}lpips": masked_lpips(self.lpips, prediction, target, mask),
        }
        for key, value in values.items():
            metrics[key].append(value)
        return values

    @torch.no_grad()
    def _eval_dataset(
        self,
        dataset,
        split: str,
        apply_train_adjustment: bool,
        artifact_writer: Optional[EvalArtifactWriter],
    ) -> Dict[str, float]:
        """Render and score one dataset split."""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0.0
        metrics = defaultdict(list)

        for data in dataloader:
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            depth_map_gt = (
                data["depth"].to(device) if "depth" in data else None
            )
            masks = data["mask"].to(device) if "mask" in data else None
            sky_masks = (
                data["sky_mask"].to(device) if "sky_mask" in data else None
            )
            if masks is not None:
                pixels = pixels * masks.unsqueeze(-1)
            image_ids = data["image_id"].to(device)
            camera_ids = data["camera_id"].to(device)
            frame_ids = data["frame_id"].to(device)
            if apply_train_adjustment and cfg.pose_opt:
                camtoworlds = self._apply_pose_adjustment(
                    camtoworlds, image_ids, frame_ids, camera_ids
                )

            radial_coeffs = (
                data["radial_coeffs"].to(device)
                if (cfg.keep_distortion and "radial_coeffs" in data)
                else None
            )
            tangential_coeffs = (
                data["tangential_coeffs"].to(device)
                if (cfg.keep_distortion and "tangential_coeffs" in data)
                else None
            )
            if cfg.calib_opt:
                if radial_coeffs is None:
                    raise RuntimeError(
                        "calib_opt requires a four-coefficient fisheye calibration"
                    )
                Ks, radial_coeffs = self._apply_calibration_adjustment(
                    Ks, radial_coeffs, camera_ids
                )
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            renders, alphas, _ = self.render_scene(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids if apply_train_adjustment else None,
                masks=masks,
                render_mode="RGB+ED",
                camera_model="fisheye" if cfg.keep_distortion and data["camera_model"] == 5 else "pinhole",
                radial_coeffs=radial_coeffs,
                tangential_coeffs=tangential_coeffs,
            )  # [1, H, W, 4]
            colors = renders[..., :3]
            depths = renders[..., 3:4]
            bilateral_colors = None
            if cfg.use_bilateral_grid and apply_train_adjustment:
                bilateral_colors = self._apply_bilateral_grid(colors, image_ids)
            ppisp_colors = None
            if cfg.use_ppisp:
                ppisp_colors = self._apply_ppisp(
                    colors,
                    camera_ids,
                    image_ids if apply_train_adjustment else None,
                )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            if bilateral_colors is not None:
                bilateral_colors = torch.clamp(bilateral_colors, 0.0, 1.0)
            if ppisp_colors is not None:
                ppisp_colors = torch.clamp(ppisp_colors, 0.0, 1.0)
            if world_rank == 0:
                image_metrics = self._append_image_metrics(
                    metrics, "", colors, pixels, masks
                )
                if bilateral_colors is not None:
                    image_metrics.update(
                        self._append_image_metrics(
                            metrics,
                            "bilateral_",
                            bilateral_colors,
                            pixels,
                            masks,
                        )
                    )
                if ppisp_colors is not None:
                    image_metrics.update(
                        self._append_image_metrics(
                            metrics,
                            "ppisp_",
                            ppisp_colors,
                            pixels,
                            masks,
                        )
                    )
                no_sky_mask = None
                if sky_masks is not None:
                    no_sky_mask = ~sky_masks
                    if masks is not None:
                        no_sky_mask = no_sky_mask & masks
                    image_metrics.update(
                        self._append_image_metrics(
                            metrics, "no_sky_", colors, pixels, no_sky_mask
                        )
                    )
                    if bilateral_colors is not None:
                        image_metrics.update(
                            self._append_image_metrics(
                                metrics,
                                "bilateral_no_sky_",
                                bilateral_colors,
                                pixels,
                                no_sky_mask,
                            )
                        )
                    if ppisp_colors is not None:
                        image_metrics.update(
                            self._append_image_metrics(
                                metrics,
                                "ppisp_no_sky_",
                                ppisp_colors,
                                pixels,
                                no_sky_mask,
                            )
                        )
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels).clamp(0.0, 1.0)
                    image_metrics.update(
                        self._append_image_metrics(
                            metrics, "cc_", cc_colors, pixels, masks
                        )
                    )
                    if no_sky_mask is not None:
                        image_metrics.update(
                            self._append_image_metrics(
                                metrics,
                                "no_sky_cc_",
                                cc_colors,
                                pixels,
                                no_sky_mask,
                            )
                        )

                if depth_map_gt is not None:
                    rendered_depth = depths.squeeze(-1)
                    depth_valid = depth_map_gt > 0.0
                    if masks is not None:
                        depth_valid &= masks
                    rendered_disp = torch.where(
                        rendered_depth > 0.0,
                        rendered_depth.clamp_min(1e-8).reciprocal(),
                        torch.zeros_like(rendered_depth),
                    )
                    target_disp = torch.where(
                        depth_valid,
                        depth_map_gt.clamp_min(1e-8).reciprocal(),
                        torch.zeros_like(depth_map_gt),
                    )
                    depth_weight = depth_valid.to(rendered_depth.dtype)
                    depth_inv_l1 = (
                        ((rendered_disp - target_disp).abs() * depth_weight).sum()
                        / depth_weight.sum().clamp_min(1.0)
                        * self.scene_scale
                    )
                    depth_valid_pixels = depth_weight.sum()
                    depth_valid_ratio = depth_weight.mean()
                    for key, value in (
                        ("depth_inv_l1", depth_inv_l1),
                        ("depth_valid_pixels", depth_valid_pixels),
                        ("depth_valid_ratio", depth_valid_ratio),
                    ):
                        metrics[key].append(value)
                        image_metrics[key] = value

                if artifact_writer is not None:
                    source_index = int(data["source_index"].item())
                    final_colors = None
                    final_label = None
                    if bilateral_colors is not None:
                        final_colors = bilateral_colors
                        final_label = "Bilateral"
                    elif ppisp_colors is not None:
                        final_colors = ppisp_colors
                        final_label = "PPISP"
                    artifact_writer.write(
                        EvalArtifact(
                            split=split,
                            image_name=self.parser.image_names[source_index],
                            image_path=self.parser.image_paths[source_index],
                            source_index=source_index,
                            split_image_id=int(image_ids.item()),
                            camera_id=int(camera_ids.item()),
                            camera_model=int(data["camera_model"].item()),
                            rig_frame_index=int(frame_ids.item()),
                            target_rgb=pixels[0].cpu().numpy(),
                            canonical_rgb=colors[0].cpu().numpy(),
                            final_rgb=(
                                final_colors[0].cpu().numpy()
                                if final_colors is not None
                                else None
                            ),
                            final_label=final_label,
                            rendered_depth=depths[0, ..., 0].cpu().numpy(),
                            target_depth=(
                                depth_map_gt[0].cpu().numpy()
                                if depth_map_gt is not None
                                else None
                            ),
                            foreground_alpha=alphas[0, ..., 0].cpu().numpy(),
                            valid_mask=(
                                masks[0].cpu().numpy() if masks is not None else None
                            ),
                            sky_mask=(
                                sky_masks[0].cpu().numpy()
                                if sky_masks is not None
                                else None
                            ),
                            metrics={
                                key: float(value.item())
                                for key, value in image_metrics.items()
                            },
                            intrinsics=Ks[0].cpu().numpy(),
                            camtoworld=camtoworlds[0].cpu().numpy(),
                            radial_coeffs=(
                                radial_coeffs[0].cpu().numpy()
                                if radial_coeffs is not None
                                else None
                            ),
                            tangential_coeffs=(
                                tangential_coeffs[0].cpu().numpy()
                                if tangential_coeffs is not None
                                else None
                            ),
                        )
                    )

        if world_rank != 0:
            return {}
        if len(dataloader) == 0:
            return {"ellipse_time": 0.0, "num_images": 0}

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats["ellipse_time"] = ellipse_time / len(dataloader)
        stats["num_images"] = len(dataloader)
        return stats

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Render validation images and a sampled subset of training images."""
        print("Running evaluation...")
        cfg = self.cfg
        world_rank = self.world_rank
        artifact_variant = "model" if stage == "val" else stage
        depth_max = self.trainset.depth_max
        if depth_max is not None:
            depth_max *= self.trainset.depth_world_scale
        else:
            depth_max = max(self.scene_scale * 4.0, 1.0)
        artifact_writer = (
            EvalArtifactWriter(
                render_dir=Path(self.render_dir),
                iteration=step,
                completed_step=step + 1,
                variant=artifact_variant,
                depth_max=depth_max,
                scene_scale=self.scene_scale,
            )
            if world_rank == 0
            else None
        )
        val_stats = {}
        if len(self.valset) > 0:
            val_stats = self._eval_dataset(
                self.valset,
                "val",
                apply_train_adjustment=False,
                artifact_writer=artifact_writer,
            )
        train_stats = self._eval_dataset(
            self.train_evalset,
            "train",
            apply_train_adjustment=True,
            artifact_writer=artifact_writer,
        )

        if world_rank == 0:
            val_image_count = val_stats.pop("num_images", 0)
            train_image_count = train_stats.pop("num_images", 0.0)
            stats = dict(val_stats)
            stats.update({f"train_{k}": v for k, v in train_stats.items()})
            stats.update(
                {
                    "num_train_images": train_image_count,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if val_image_count:
                stats["num_val_images"] = val_image_count
            if self.sky_splats is not None:
                stats["num_sky_GS"] = len(self.sky_splats["means"])

            if not val_image_count and cfg.use_bilateral_grid:
                print(
                    f"Train full PSNR: {stats['train_psnr']:.3f}, SSIM: {stats['train_ssim']:.4f}, LPIPS: {stats['train_lpips']:.3f}, "
                    f"CC_PSNR: {stats['train_cc_psnr']:.3f}, CC_SSIM: {stats['train_cc_ssim']:.4f}, CC_LPIPS: {stats['train_cc_lpips']:.3f}; "
                    f"Train time: {stats['train_ellipse_time']:.3f}s/image, "
                    f"Number of GS: {stats['num_GS']}"
                )
            elif not val_image_count and cfg.use_ppisp:
                print(
                    f"Train canonical PSNR: {stats['train_psnr']:.3f}, SSIM: {stats['train_ssim']:.4f}, LPIPS: {stats['train_lpips']:.3f}; "
                    f"PPISP PSNR: {stats['train_ppisp_psnr']:.3f}, SSIM: {stats['train_ppisp_ssim']:.4f}, LPIPS: {stats['train_ppisp_lpips']:.3f}; "
                    f"Train time: {stats['train_ellipse_time']:.3f}s/image, "
                    f"Number of GS: {stats['num_GS']}"
                )
            elif not val_image_count:
                print(
                    f"Train full PSNR: {stats['train_psnr']:.3f}, SSIM: {stats['train_ssim']:.4f}, LPIPS: {stats['train_lpips']:.3f}; "
                    f"Train time: {stats['train_ellipse_time']:.3f}s/image, "
                    f"Number of GS: {stats['num_GS']}"
                )
            elif cfg.use_bilateral_grid:
                print(
                    f"Val full PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}, "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f}; "
                    f"Train full PSNR: {stats['train_psnr']:.3f}, SSIM: {stats['train_ssim']:.4f}, LPIPS: {stats['train_lpips']:.3f}, "
                    f"CC_PSNR: {stats['train_cc_psnr']:.3f}, CC_SSIM: {stats['train_cc_ssim']:.4f}, CC_LPIPS: {stats['train_cc_lpips']:.3f}; "
                    f"Val time: {stats['ellipse_time']:.3f}s/image, "
                    f"Train time: {stats['train_ellipse_time']:.3f}s/image, "
                    f"Number of GS: {stats['num_GS']}"
                )
            elif cfg.use_ppisp:
                print(
                    f"Val canonical PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}; "
                    f"Val PPISP PSNR: {stats['ppisp_psnr']:.3f}, SSIM: {stats['ppisp_ssim']:.4f}, LPIPS: {stats['ppisp_lpips']:.3f}; "
                    f"Train canonical PSNR: {stats['train_psnr']:.3f}, SSIM: {stats['train_ssim']:.4f}, LPIPS: {stats['train_lpips']:.3f}; "
                    f"Train PPISP PSNR: {stats['train_ppisp_psnr']:.3f}, SSIM: {stats['train_ppisp_ssim']:.4f}, LPIPS: {stats['train_ppisp_lpips']:.3f}; "
                    f"Val time: {stats['ellipse_time']:.3f}s/image, "
                    f"Train time: {stats['train_ellipse_time']:.3f}s/image, "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"Val full PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}; "
                    f"Train full PSNR: {stats['train_psnr']:.3f}, SSIM: {stats['train_ssim']:.4f}, LPIPS: {stats['train_lpips']:.3f}; "
                    f"Val time: {stats['ellipse_time']:.3f}s/image, "
                    f"Train time: {stats['train_ellipse_time']:.3f}s/image, "
                    f"Number of GS: {stats['num_GS']}"
                )
            if "no_sky_psnr" in stats:
                print(
                    f"Val no-sky PSNR: {stats['no_sky_psnr']:.3f}, "
                    f"SSIM: {stats['no_sky_ssim']:.4f}, "
                    f"LPIPS: {stats['no_sky_lpips']:.3f}; "
                    f"Train no-sky PSNR: {stats['train_no_sky_psnr']:.3f}, "
                    f"SSIM: {stats['train_no_sky_ssim']:.4f}, "
                    f"LPIPS: {stats['train_no_sky_lpips']:.3f}"
                )
            elif "train_no_sky_psnr" in stats:
                print(
                    f"Train no-sky PSNR: {stats['train_no_sky_psnr']:.3f}, "
                    f"SSIM: {stats['train_no_sky_ssim']:.4f}, "
                    f"LPIPS: {stats['train_no_sky_lpips']:.3f}"
                )
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            if artifact_writer is not None:
                manifest_path = artifact_writer.finalize(stats)
                print(f"Evaluation artifacts saved to {manifest_path}")
            self.train_logger.log("evaluation", step=step, stage=stage, **stats)

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]
        camera_id = next(iter(self.parser.Ks_dict))
        is_fisheye = (
            cfg.keep_distortion
            and self.parser.camera_models_dict[camera_id] == 5
        )
        radial_coeffs = (
            torch.from_numpy(self.parser.params_dict[camera_id])[None].float().to(device)
            if is_fisheye
            else None
        )
        if cfg.calib_opt:
            if radial_coeffs is None:
                raise RuntimeError(
                    "calib_opt requires a four-coefficient fisheye calibration"
                )
            K_batch, radial_coeffs = self._apply_calibration_adjustment(
                K[None],
                radial_coeffs,
                torch.tensor([camera_id], device=device),
            )
            K = K_batch[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.render_scene(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                camera_model="fisheye" if is_fisheye else "pinhole",
                radial_coeffs=radial_coeffs,
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        camera_model = render_tab_state.camera_model

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.render_scene(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=camera_model
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if runner.sky_splats is not None:
            if any("sky_splats" not in ckpt for ckpt in ckpts):
                raise ValueError(
                    "The selected checkpoint predates Gaussian sky state; "
                    "evaluate it with --no-sky-enabled or use a sky checkpoint."
                )
            for k in runner.sky_splats.keys():
                runner.sky_splats[k].data = torch.cat(
                    [ckpt["sky_splats"][k] for ckpt in ckpts]
                )
        if cfg.pose_opt:
            if "pose_adjust" not in ckpts[0]:
                raise ValueError(
                    "pose_opt evaluation requires pose_adjust in the checkpoint"
                )
            runner._pose_module().load_state_dict(ckpts[0]["pose_adjust"])
        if cfg.calib_opt:
            if "calibration_adjust" not in ckpts[0]:
                raise ValueError(
                    "calib_opt evaluation requires calibration_adjust in the checkpoint"
                )
            runner._calibration_module().load_state_dict(
                ckpts[0]["calibration_adjust"]
            )
        if cfg.use_bilateral_grid:
            if "bilateral_grid" not in ckpts[0]:
                raise ValueError(
                    "use_bilateral_grid evaluation requires bilateral_grid in the checkpoint"
                )
            runner.bil_grids.load_state_dict(ckpts[0]["bilateral_grid"])
        if cfg.use_ppisp:
            if "ppisp" not in ckpts[0]:
                raise ValueError(
                    "use_ppisp evaluation requires PPISP state in the checkpoint"
                )
            saved_camera_ids = tuple(
                int(camera_id)
                for camera_id in ckpts[0].get("ppisp_camera_ids", ())
            )
            if saved_camera_ids != runner.ppisp_camera_ids:
                raise ValueError(
                    "Checkpoint PPISP camera ids do not match the current dataset"
                )
            saved_frame_indices = ckpts[0].get("ppisp_frame_indices")
            if saved_frame_indices is None or not torch.equal(
                saved_frame_indices.cpu(),
                runner.ppisp_frame_indices.cpu(),
            ):
                raise ValueError(
                    "Checkpoint PPISP frame indices do not match the current train split"
                )
            runner.ppisp.load_state_dict(ckpts[0]["ppisp"])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Load YAML; later CLI arguments override YAML values.
    CUDA_VISIBLE_DEVICES=9 python examples/simple_trainer.py --config configs/simple_trainer/park.yaml

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    cfg = parse_config()
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut or cfg.with_geer:
        assert cfg.with_eval3d, "Training with UT or GEER requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
