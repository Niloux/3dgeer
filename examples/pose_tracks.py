"""Persistent SfM reprojection anchors, independent of Gaussian primitives."""

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch import Tensor, nn


def project_points(
    points_cam: Tensor,
    K: Tensor,
    camera_model: str,
    distortion: Tensor,
) -> Tensor:
    """Project points with the parser's OpenCV camera convention.

    Fisheye coefficients are k1..k4; pinhole coefficients are k1,k2,p1,p2.
    Coordinates use COLMAP's corner origin, with pixel centers at i + 0.5.
    """
    xy, z = points_cam[..., :2], points_cam[..., 2]
    if camera_model == "fisheye":
        # Clamping the norm, rather than sqrt(x*x+y*y), keeps the optical-axis
        # backward finite and preserves the limiting projection Jacobian.
        radius = torch.linalg.vector_norm(xy, dim=-1).clamp_min(1e-8)
        theta = torch.atan2(radius, z)
        theta2 = theta.square()
        k1, k2, k3, k4 = distortion.unbind(-1)
        theta_d = theta * (
            1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
        )
        projected = xy * (theta_d / radius).unsqueeze(-1)
    elif camera_model == "pinhole":
        x, y = (xy / z.clamp_min(1e-8).unsqueeze(-1)).unbind(-1)
        k1, k2, p1, p2 = distortion.unbind(-1)
        r2 = x.square() + y.square()
        radial = 1 + r2 * (k1 + r2 * k2)
        projected = torch.stack(
            (
                x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x.square()),
                y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y.square()),
            ),
            dim=-1,
        )
    else:
        raise ValueError(f"Unsupported track camera model: {camera_model}")
    return projected * K.diagonal()[:2] + K[:2, 2]


class PoseTrackAnchors(nn.Module):
    """Shared, trainable 3D track points and fixed training-image observations.

    Only training observations participate in filtering and optimization. The
    feature measurements and point identities survive Gaussian densification.
    Full, original (possibly distorted) images are required.
    """

    def __init__(
        self,
        parser,
        train_indices,
        camera_model: str,
        min_views: int = 3,
        max_reprojection_error: float = 2.0,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        if parser.undistort:
            raise ValueError("Pose track anchors require keep_distortion=True")
        if parser.track_observations is None:
            raise ValueError("Parser must retain feature tracks with load_tracks=True")
        self.camera_model = camera_model
        self.huber_delta = huber_delta
        self.image_names = [parser.image_names[int(i)] for i in train_indices]
        records = []
        counts = np.zeros(len(parser.points), dtype=np.int32)
        base_sizes = []
        distortions = []
        initial_errors = []
        for source_index, name in zip(train_indices, self.image_names):
            camera_id = parser.camera_ids[int(source_index)]
            expected_model = (
                "fisheye" if parser.camera_models_dict[camera_id] == 5 else "pinhole"
            )
            if camera_model != expected_model:
                raise ValueError(
                    f"Track camera {camera_id} is {expected_model}, "
                    f"but the renderer uses {camera_model}"
                )
            base_size = np.asarray(parser._base_imsize_dict[camera_id])
            size = np.asarray(parser.imsize_dict[camera_id])
            base_sizes.append(base_size)
            distortion = np.zeros(4, dtype=np.float64)
            params = parser.params_dict[camera_id]
            distortion[:len(params)] = params
            distortions.append(distortion)
            indices = parser.point_indices.get(name, np.empty(0, dtype=np.int32))
            xy = parser.track_observations.get(name, np.empty((0, 2)))
            # Count distinct views, even if an input track repeats an image.
            indices, first = np.unique(indices, return_index=True)
            xy = xy[first]
            scaled_xy = xy * (size / base_size)
            c2w = parser.camtoworlds[int(source_index)]
            points_cam = (
                parser.points[indices].astype(np.float64) - c2w[:3, 3]
            ) @ c2w[:3, :3]
            projected = project_points(
                torch.from_numpy(points_cam),
                torch.from_numpy(parser.Ks_dict[camera_id]),
                camera_model,
                torch.from_numpy(distortion),
            ).numpy()
            errors = np.linalg.norm(projected - scaled_xy, axis=-1)
            valid = (
                np.isfinite(errors)
                & (points_cam[:, 2] > 0)
                & (errors <= max_reprojection_error)
                & (scaled_xy >= 0).all(-1)
                & (scaled_xy < size).all(-1)
            )
            # Apply fixed FOV, dataset and sky masks before counting track views.
            mask = parser.mask_dict[camera_id]
            mask = (
                np.ones(tuple(size[::-1]), dtype=bool)
                if mask is None else mask.copy()
            )
            for path, is_sky in (
                (parser.mask_paths[int(source_index)], False),
                (parser.sky_mask_paths[int(source_index)], True),
            ):
                if path is not None:
                    pixels = imageio.imread(path)
                    if pixels.ndim == 3:
                        pixels = pixels[..., 0]
                    pixels = cv2.resize(
                        pixels, tuple(size), interpolation=cv2.INTER_NEAREST
                    ) > 127
                    mask &= ~pixels if is_sky else pixels
            candidates = np.flatnonzero(valid)
            ix, iy = np.floor(scaled_xy[candidates]).astype(np.int64).T
            valid[candidates] &= mask[iy, ix]
            records.append((indices[valid], xy[valid], errors[valid]))
            counts[indices[valid]] += 1

        selected = np.flatnonzero(counts >= min_views)
        if not len(selected):
            raise ValueError(
                "No SfM tracks survive training-view, mask and reprojection filtering; "
                "check the camera model, feature measurements and pose_track_* thresholds"
            )
        remap = np.full(len(counts), -1, dtype=np.int64)
        remap[selected] = np.arange(len(selected))
        point_indices, observations, offsets = [], [], [0]
        for indices, xy, errors in records:
            keep = remap[indices] >= 0
            point_indices.append(remap[indices[keep]])
            observations.append(xy[keep])
            initial_errors.append(errors[keep])
            offsets.append(offsets[-1] + int(keep.sum()))

        # from_pretrained avoids consuming the trainer's random-number stream.
        self.points = nn.Embedding.from_pretrained(
            torch.from_numpy(parser.points[selected].copy()), freeze=False, sparse=True
        )
        self.register_buffer(
            "point_ids", torch.from_numpy(parser.track_point_ids[selected].copy())
        )
        self.register_buffer(
            "observation_point_indices", torch.from_numpy(np.concatenate(point_indices))
        )
        self.register_buffer(
            "observation_xy", torch.from_numpy(np.concatenate(observations)).float()
        )
        self.register_buffer("image_offsets", torch.tensor(offsets, dtype=torch.int64))
        self.register_buffer(
            "base_image_sizes", torch.tensor(np.asarray(base_sizes), dtype=torch.float32)
        )
        self.register_buffer(
            "distortions", torch.tensor(np.asarray(distortions), dtype=torch.float32)
        )
        self._offsets = offsets  # CPU slicing avoids a device synchronization per step.
        errors = np.concatenate(initial_errors)
        self.summary = {
            "num_tracks": len(selected),
            "num_observations": len(errors),
            "num_images": len(records),
            "images_without_tracks": [
                name for i, name in enumerate(self.image_names)
                if offsets[i] == offsets[i + 1]
            ],
            "initial_reprojection_mean_px": float(errors.mean()),
            "initial_reprojection_median_px": float(np.median(errors)),
            "initial_reprojection_p90_px": float(np.quantile(errors, 0.9)),
            "filter_data_factor": parser.factor,
        }

    def forward(
        self,
        image_id: int,
        camtoworld: Tensor,
        K: Tensor,
        width: int,
        height: int,
        compute_stats: bool = False,
    ) -> tuple[Tensor, dict]:
        start, end = self._offsets[image_id : image_id + 2]
        if start == end:
            return camtoworld.sum() * 0.0, {"observations": 0}
        points = self.points(self.observation_point_indices[start:end])
        points_cam = (points - camtoworld[:3, 3]) @ camtoworld[:3, :3]
        predicted = project_points(
            points_cam, K, self.camera_model, self.distortions[image_id]
        )
        scale = K.new_tensor([width, height]) / self.base_image_sizes[image_id]
        target = self.observation_xy[start:end] * scale
        residual = torch.linalg.vector_norm(predicted - target, dim=-1)
        # Radial Huber in training pixels, averaged over this image's tracks.
        # Do not filter large current residuals: doing so would let poses escape
        # the very constraints that are intended to resist drift.
        quadratic = residual.clamp_max(self.huber_delta)
        loss = (
            0.5 * quadratic.square() + self.huber_delta * (residual - quadratic)
        ).mean()
        stats = {}
        if compute_stats:
            error = residual.detach()
            stats = {
                "observations": end - start,
                "reprojection_mean_px": error.mean(),
                "reprojection_median_px": error.median(),
                "reprojection_p90_px": torch.quantile(error, 0.9),
                "behind_camera_fraction": (
                    points_cam.detach()[:, 2] <= 0
                ).float().mean(),
            }
        return loss, stats
