import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resolve_sky_mask_path(mask_dir: str, image_name: str) -> str:
    """Resolve nested COLMAP names against flat or nested PNG sky-mask folders."""
    stem = os.path.splitext(image_name)[0]
    candidates = [
        os.path.join(mask_dir, image_name + ".png"),
        os.path.join(mask_dir, stem + ".png"),
        os.path.join(mask_dir, os.path.basename(stem) + ".png"),
    ]
    for path in dict.fromkeys(candidates):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Sky mask for COLMAP image {image_name!r} was not found in {mask_dir!r}"
    )


def _resolve_depth_path(depth_dir: str, image_name: str) -> str:
    """Resolve an image name to its same-layout PNG depth sidecar."""
    relative = os.path.splitext(image_name)[0] + ".png"
    path = os.path.join(depth_dir, relative)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Depth map for COLMAP image {image_name!r} was not found at {path!r}"
        )
    return path


def _infer_lfs_depth_max(data_dir: str) -> float:
    """Read the uint16 depth encoding range emitted by the preprocessing pipeline."""
    report_path = os.path.join(data_dir, "report.json")
    if not os.path.isfile(report_path):
        raise ValueError(
            "depth_max must be provided when the dataset has no report.json"
        )
    with open(report_path) as stream:
        report = json.load(stream)
    candidates = (
        report.get("lidar_initialization", {})
        .get("supervision_maps", {})
        .get("depth", {})
        .get("z_max_m"),
        report.get("settings", {}).get("lidar_max_camera_distance_m"),
    )
    depth_max = next((value for value in candidates if value is not None), None)
    if depth_max is None:
        raise ValueError(
            f"Cannot infer depth_max from {report_path}; set it explicitly"
        )
    depth_max = float(depth_max)
    if not math.isfinite(depth_max) or depth_max <= 0.0:
        raise ValueError(f"depth_max must be finite and positive, got {depth_max}")
    return depth_max


def _downsample_sparse_depth(
    encoded: np.ndarray, output_shape: tuple[int, int]
) -> np.ndarray:
    """Downsample a sparse z-buffer while retaining the nearest valid sample."""
    output_height, output_width = output_shape
    if encoded.shape == output_shape:
        return encoded

    source_height, source_width = encoded.shape
    if (
        source_height < output_height
        or source_width < output_width
        or source_height % output_height != 0
        or source_width % output_width != 0
        or source_height // output_height != source_width // output_width
    ):
        raise ValueError(
            "Depth and training image sizes must differ by the same integer "
            f"factor, got {encoded.shape} and {output_shape}"
        )

    ys, xs = np.nonzero(encoded)
    output = np.zeros(output_height * output_width, dtype=np.uint16)
    if len(xs) == 0:
        return output.reshape(output_shape)

    output_xs = np.floor((xs + 0.5) * output_width / source_width).astype(
        np.int64
    )
    output_ys = np.floor((ys + 0.5) * output_height / source_height).astype(
        np.int64
    )
    np.clip(output_xs, 0, output_width - 1, out=output_xs)
    np.clip(output_ys, 0, output_height - 1, out=output_ys)
    output_indices = output_ys * output_width + output_xs

    # uint16 value 65535 is valid, so use a wider sentinel while reducing.
    nearest = np.full(output.size, 65536, dtype=np.uint32)
    np.minimum.at(nearest, output_indices, encoded[ys, xs].astype(np.uint32))
    valid = nearest <= 65535
    output[valid] = nearest[valid].astype(np.uint16)
    return output.reshape(output_shape)


def _frame_key(image_name: str):
    """Return a rig frame key for names such as ``*_00063-L.png``."""
    match = re.search(r"_(\d+)-[A-Za-z](?:\.[^.]+)?$", image_name)
    return ("frame", int(match.group(1))) if match else ("image", image_name)


def _combine_supervision_masks(
    camera_mask: Optional[np.ndarray],
    dataset_mask: Optional[np.ndarray],
    sky_mask: Optional[np.ndarray],
):
    """Restore sky supervision without re-enabling invalid camera pixels."""
    valid = None if camera_mask is None else camera_mask.astype(bool, copy=False)
    dataset = None if dataset_mask is None else dataset_mask.astype(bool, copy=False)
    sky = None if sky_mask is None else sky_mask.astype(bool, copy=False)
    if dataset is not None:
        supervised = dataset if sky is None else dataset | sky
        valid = supervised if valid is None else valid & supervised
    if sky is not None and valid is not None:
        sky = sky & valid
    return valid, sky


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


def _fisheye_fov_mask(K, params, width: int, height: int, max_fov: float):
    if not 0.0 < max_fov < 180.0:
        raise ValueError("max_fisheye_fov must be between 0 and 180 degrees")
    theta = math.radians(max_fov / 2.0)
    theta2 = theta * theta
    k1, k2, k3, k4 = params
    radius = theta * (
        1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
    )
    ys, xs = np.ogrid[:height, :width]
    return (
        ((xs + 0.5 - K[0, 2]) / K[0, 0]) ** 2
        + ((ys + 0.5 - K[1, 2]) / K[1, 1]) ** 2
        < radius**2
    )


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        undistort: bool = True,
        max_fisheye_fov: Optional[float] = None,
        frame_id_min: Optional[int] = None,
        frame_id_max: Optional[int] = None,
        sky_mask_dir: Optional[str] = None,
        use_test_split: bool = True,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        if test_every <= 0:
            raise ValueError("test_every must be positive")
        self.test_every = test_every
        self.use_test_split = use_test_split
        self.undistort = undistort
        self.max_fisheye_fov = max_fisheye_fov
        if sky_mask_dir is not None and not os.path.isabs(sky_mask_dir):
            sky_mask_dir = os.path.join(data_dir, sky_mask_dir)
        if sky_mask_dir is not None and not os.path.isdir(sky_mask_dir):
            raise FileNotFoundError(f"Sky mask directory does not exist: {sky_mask_dir}")
        self.sky_mask_dir = sky_mask_dir
        if max_fisheye_fov is not None and undistort:
            raise ValueError("max_fisheye_fov requires undistort=False")

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        camera_models_dict = dict()
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camera_models_dict[camera_id] = 0
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camera_models_dict[camera_id] = 1
            elif type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camera_models_dict[camera_id] = 2
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camera_models_dict[camera_id] = 3
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camera_models_dict[camera_id] = 4
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camera_models_dict[camera_id] = 5
            else:
                raise ValueError(f"Unsupported camera type: {type_}")

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        if frame_id_min is not None or frame_id_max is not None:
            def in_frame_range(name: str) -> bool:
                match = re.search(r"_(\d+)-[LR](?:\.[^.]+)?$", name)
                if match is None:
                    raise ValueError(
                        f"Cannot parse frame id from COLMAP image name {name!r}"
                    )
                frame_id = int(match.group(1))
                return (
                    (frame_id_min is None or frame_id >= frame_id_min)
                    and (frame_id_max is None or frame_id <= frame_id_max)
                )

            frame_keep = np.array([in_frame_range(name) for name in image_names])
            image_names = [name for name, keep in zip(image_names, frame_keep) if keep]
            camtoworlds = camtoworlds[frame_keep]
            camera_ids = [camera_id for camera_id, keep in zip(camera_ids, frame_keep) if keep]
            if not image_names:
                raise ValueError(
                    f"No COLMAP images found in frame range {frame_id_min}..{frame_id_max}."
                )
            print(
                f"[Parser] frame filter kept {len(image_names)} images "
                f"({frame_id_min}..{frame_id_max}, inclusive)."
            )

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        mask_dir = os.path.join(data_dir, "masks")
        has_masks = os.path.isdir(mask_dir)
        if not os.path.exists(colmap_image_dir):
            raise ValueError(f"Image folder {colmap_image_dir} does not exist.")
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        if not os.path.exists(image_dir):
            if factor == 1:
                raise ValueError(f"Image folder {image_dir} does not exist.")
            image_dir = _resize_image_folder(colmap_image_dir, image_dir, factor)
        elif factor > 1 and len(_get_rel_paths(image_dir)) < len(colmap_files):
            image_dir = _resize_image_folder(colmap_image_dir, image_dir, factor)

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))

        # Skip images referenced by COLMAP but missing on disk.
        kept_image_names = []
        kept_image_paths = []
        kept_mask_paths = []
        kept_sky_mask_paths = []
        kept_camera_ids = []
        kept_camtoworlds = []
        skipped = []

        for name, camera_id, c2w in zip(image_names, camera_ids, camtoworlds):
            rel_path = name if name in image_files else colmap_to_image.get(name, None)
            if rel_path is None or rel_path not in image_files:
                skipped.append(name)
                continue
            kept_image_names.append(name)
            kept_image_paths.append(os.path.join(image_dir, rel_path))
            if has_masks:
                mask_path = os.path.join(mask_dir, name + ".png")
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(mask_path)
                kept_mask_paths.append(mask_path)
            else:
                kept_mask_paths.append(None)
            if self.sky_mask_dir is not None:
                kept_sky_mask_paths.append(
                    _resolve_sky_mask_path(self.sky_mask_dir, name)
                )
            else:
                kept_sky_mask_paths.append(None)
            kept_camera_ids.append(camera_id)
            kept_camtoworlds.append(c2w)

        if skipped:
            print(
                f"Warning: skipped {len(skipped)} COLMAP images not found on disk "
                f"(e.g. {skipped[:5]})."
            )
        if len(kept_image_names) == 0:
            raise ValueError(
                f"No COLMAP images could be mapped to files in {image_dir!r}."
            )

        image_names = kept_image_names
        image_paths = kept_image_paths
        camera_ids = kept_camera_ids
        camtoworlds = np.stack(kept_camtoworlds, axis=0)

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items() if k in image_names
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.mask_paths = kept_mask_paths  # List[Optional[str]], (num_images,)
        self.sky_mask_paths = kept_sky_mask_paths
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        frame_keys = [_frame_key(name) for name in image_names]
        frame_key_to_index = {
            key: index for index, key in enumerate(dict.fromkeys(frame_keys))
        }
        self.frame_ids = np.asarray(
            [frame_key_to_index[key] for key in frame_keys], dtype=np.int64
        )
        self.camera_models_dict = camera_models_dict  # Dict of camera_id -> camera_model (int)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_dict = dict()
        for camera_id in self.params_dict.keys():
            if self.undistort:
                camtype = "perspective" if self.camera_models_dict[camera_id] != 5 else "fisheye"
                params = self.params_dict[camera_id]
                if len(params) == 0:
                    continue  # no distortion
                assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
                assert (
                    camera_id in self.params_dict
                ), f"Missing params for camera {camera_id}"
                K = self.Ks_dict[camera_id]
                width, height = self.imsize_dict[camera_id]

                if camtype == "perspective":
                    K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                        K, params, (width, height), 0
                    )
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        K, params, None, K_undist, (width, height), cv2.CV_32FC1
                    )
                    mask = None
                elif camtype == "fisheye":
                    D = params.astype(np.float64).reshape(4, 1)
                    R = np.eye(3, dtype=np.float64)
                    K_undist = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                        K.astype(np.float64), D, (width, height), R, balance=0.0
                    )
                    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
                        K.astype(np.float64),
                        D,
                        R,
                        K_undist.astype(np.float64),
                        (width, height),
                        cv2.CV_32FC1,
                    )

                    valid = (
                        np.isfinite(mapx)
                        & np.isfinite(mapy)
                        & (mapx >= 0)
                        & (mapx <= float(width - 1))
                        & (mapy >= 0)
                        & (mapy <= float(height - 1))
                    )
                    ys, xs = np.where(valid)
                    if xs.size == 0:
                        roi_undist = [0, 0, width, height]
                        mask = None
                    else:
                        x_min, y_min = int(xs.min()), int(ys.min())
                        x_max, y_max = int(xs.max()) + 1, int(ys.max()) + 1
                        mapx = mapx[y_min:y_max, x_min:x_max]
                        mapy = mapy[y_min:y_max, x_min:x_max]
                        mask = valid[y_min:y_max, x_min:x_max]
                        K_undist = K_undist.copy()
                        K_undist[0, 2] -= float(x_min)
                        K_undist[1, 2] -= float(y_min)
                        roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
                else:
                    assert_never(camtype)

                self.mapx_dict[camera_id] = mapx
                self.mapy_dict[camera_id] = mapy
                self.Ks_dict[camera_id] = K_undist
                self.roi_dict[camera_id] = roi_undist
                self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
                self.mask_dict[camera_id] = mask
            else:
                self.roi_dict[camera_id] = [0, 0, *self.imsize_dict[camera_id]]

        if self.max_fisheye_fov is not None:
            for camera_id, camera_model in self.camera_models_dict.items():
                if camera_model != 5:
                    continue
                width, height = self.imsize_dict[camera_id]
                fov_mask = _fisheye_fov_mask(
                    self.Ks_dict[camera_id],
                    self.params_dict[camera_id],
                    width,
                    height,
                    self.max_fisheye_fov,
                )
                mask = self.mask_dict[camera_id]
                self.mask_dict[camera_id] = fov_mask if mask is None else mask & fov_mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        depth_dir: Optional[str] = None,
        depth_max: Optional[float] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.depth_paths: Optional[List[str]] = None
        self.depth_max = None
        self.depth_world_scale = 1.0
        if load_depths and depth_dir is not None:
            if parser.undistort:
                raise ValueError(
                    "Depth sidecars are in the original distorted image domain; "
                    "set undistort=False (simple_trainer: keep_distortion=true)"
                )
            if not os.path.isabs(depth_dir):
                depth_dir = os.path.join(parser.data_dir, depth_dir)
            if not os.path.isdir(depth_dir):
                raise FileNotFoundError(f"Depth directory does not exist: {depth_dir}")
            self.depth_paths = [
                _resolve_depth_path(depth_dir, name) for name in parser.image_names
            ]
            self.depth_max = (
                _infer_lfs_depth_max(parser.data_dir)
                if depth_max is None
                else float(depth_max)
            )
            if not math.isfinite(self.depth_max) or self.depth_max <= 0.0:
                raise ValueError(
                    f"depth_max must be finite and positive, got {self.depth_max}"
                )
            transform_scale = np.linalg.norm(parser.transform[:3, :3], axis=0)
            if not np.allclose(transform_scale, transform_scale[0], rtol=1e-5):
                raise ValueError("Depth maps require a similarity world transform")
            self.depth_world_scale = float(transform_scale.mean())
            print(
                f"[Dataset] Loading Z-depth maps from {depth_dir}; "
                f"uint16 range={self.depth_max:g} m"
            )
        indices = np.arange(len(self.parser.image_names))
        if not self.parser.use_test_split:
            self.indices = indices if split == "train" else indices[:0]
        elif split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        depth = None
        if self.depth_paths is not None:
            encoded_depth = cv2.imread(
                self.depth_paths[index], cv2.IMREAD_UNCHANGED
            )
            if encoded_depth is None:
                raise OSError(f"Failed to read depth map: {self.depth_paths[index]}")
            if encoded_depth.dtype != np.uint16 or encoded_depth.ndim != 2:
                raise ValueError(
                    f"Expected a single-channel uint16 depth PNG, got "
                    f"dtype={encoded_depth.dtype}, shape={encoded_depth.shape}: "
                    f"{self.depth_paths[index]}"
                )
            encoded_depth = _downsample_sparse_depth(
                encoded_depth, image.shape[:2]
            )
            valid_depth = encoded_depth > 0
            depth = np.zeros(encoded_depth.shape, dtype=np.float32)
            depth[valid_depth] = (
                (encoded_depth[valid_depth].astype(np.float32) - 1.0)
                / 65534.0
                * self.depth_max
                * self.depth_world_scale
            )
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]
        mask_path = self.parser.mask_paths[index]
        sky_mask_path = self.parser.sky_mask_paths[index]
        dataset_mask = None
        if mask_path is not None:
            dataset_mask = imageio.imread(mask_path)
            if dataset_mask.ndim == 3:
                dataset_mask = dataset_mask[..., 0]
            if dataset_mask.shape != image.shape[:2]:
                dataset_mask = cv2.resize(
                    dataset_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            dataset_mask = dataset_mask > 127
        sky_mask = None
        if sky_mask_path is not None:
            sky_mask = imageio.imread(sky_mask_path)
            if sky_mask.ndim == 3:
                sky_mask = sky_mask[..., 0]
            if sky_mask.shape != image.shape[:2]:
                sky_mask = cv2.resize(
                    sky_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            sky_mask = sky_mask > 127

        if self.parser.undistort and len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            if dataset_mask is not None:
                dataset_mask = cv2.remap(
                    dataset_mask.astype(np.uint8),
                    mapx,
                    mapy,
                    cv2.INTER_NEAREST,
                ).astype(bool)
            if sky_mask is not None:
                sky_mask = cv2.remap(
                    sky_mask.astype(np.uint8),
                    mapx,
                    mapy,
                    cv2.INTER_NEAREST,
                ).astype(bool)
            x, y, w, h = self.parser.roi_dict[camera_id]
            image = image[y : y + h, x : x + w]
            if dataset_mask is not None:
                dataset_mask = dataset_mask[y : y + h, x : x + w]
            if sky_mask is not None:
                sky_mask = sky_mask[y : y + h, x : x + w]

        mask, sky_mask = _combine_supervision_masks(mask, dataset_mask, sky_mask)

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]
            if sky_mask is not None:
                sky_mask = sky_mask[
                    y : y + self.patch_size, x : x + self.patch_size
                ]
            if depth is not None:
                depth = depth[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "camera_model": self.parser.camera_models_dict[camera_id],
            "camera_id": camera_id,
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "source_index": int(index),  # stable index in parser.image_names
            "frame_id": int(self.parser.frame_ids[index]),
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()
        if sky_mask is not None:
            data["sky_mask"] = torch.from_numpy(sky_mask).bool()

        if not self.parser.undistort:
            # Provide distortion coefficients for renderers that support it.
            if self.parser.camera_models_dict[camera_id] == 5:  # fisheye
                data["radial_coeffs"] = torch.from_numpy(params.copy()).float()
            else:
                k1, k2, p1, p2 = (
                    float(params[0]) if len(params) > 0 else 0.0,
                    float(params[1]) if len(params) > 1 else 0.0,
                    float(params[2]) if len(params) > 2 else 0.0,
                    float(params[3]) if len(params) > 3 else 0.0,
                )
                data["radial_coeffs"] = torch.tensor(
                    [k1, k2, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
                )
                data["tangential_coeffs"] = torch.tensor(
                    [p1, p2], dtype=torch.float32
                )

        if self.load_depths:
            if depth is not None:
                data["depth"] = torch.from_numpy(depth).float()
            else:
                # Projected COLMAP tracks provide the legacy sparse-depth path.
                worldtocams = np.linalg.inv(camtoworlds)
                image_name = self.parser.image_names[index]
                point_indices = self.parser.point_indices[image_name]
                points_world = self.parser.points[point_indices]
                points_cam = (
                    worldtocams[:3, :3] @ points_world.T
                    + worldtocams[:3, 3:4]
                ).T
                points_proj = (K @ points_cam.T).T
                points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
                depths = points_cam[:, 2]  # (M,)
                # filter out points outside the image
                selector = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < image.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < image.shape[0])
                    & (depths > 0)
                )
                points = points[selector]
                depths = depths[selector]
                data["points"] = torch.from_numpy(points).float()
                data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
