"""Persistent SfM tracks for GloSplat-style joint photometric/geometric training."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TrackReprojection(nn.Module):
    """Optimize track points independently of Gaussian birth, movement and pruning."""

    def __init__(self, points: Tensor, point_indices: Tensor):
        super().__init__()
        self.points = nn.Parameter(points.clone())
        self.register_buffer("point_indices", point_indices.clone())

    def forward(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        track_ids: Tensor,
        observations: Tensor,
        batch_ids: Tensor,
        camera_model: str,
        radial_coeffs: Tensor | None = None,
        tangential_coeffs: Tensor | None = None,
        huber_delta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        worldtocams = torch.linalg.inv(camtoworlds)
        views = worldtocams[batch_ids]
        points = self.points[track_ids]
        camera_points = (views[:, :3, :3] @ points[..., None]).squeeze(-1) + views[:, :3, 3]
        # Keep the empty-observation case connected to both parameter tables,
        # including on distributed workers that happen to sample no tracks.
        valid = (camera_points[:, 2] > 1e-6) & torch.isfinite(camera_points).all(-1)
        camera_points = camera_points[valid]
        batch_ids = batch_ids[valid]
        observations = observations[valid]
        x, y, z = camera_points.unbind(-1)
        if camera_model == "fisheye":
            # OPENCV_FISHEYE: theta * (1 + k1 theta² + ... + k4 theta⁸).
            radius = (x.square() + y.square()).clamp_min(1e-12).sqrt()
            theta = torch.atan2(radius, z)
            theta2 = theta.square()
            distortion = torch.ones_like(theta)
            if radial_coeffs is not None:
                k1, k2, k3, k4 = radial_coeffs[batch_ids].unbind(-1)
                distortion = 1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
            scale = theta * distortion / radius
            xy = torch.stack((x * scale, y * scale), dim=-1)
        else:
            x, y = x / z, y / z
            r2 = x.square() + y.square()
            radial = torch.ones_like(x)
            if radial_coeffs is not None:
                k1, k2, k3, k4, k5, k6 = radial_coeffs[batch_ids].unbind(-1)
                radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (
                    1 + r2 * (k4 + r2 * (k5 + r2 * k6))
                )
            xd, yd = x * radial, y * radial
            if tangential_coeffs is not None:
                p1, p2 = tangential_coeffs[batch_ids].unbind(-1)
                xd = xd + 2 * p1 * x * y + p2 * (r2 + 2 * x.square())
                yd = yd + p1 * (r2 + 2 * y.square()) + 2 * p2 * x * y
            xy = torch.stack((xd, yd), dim=-1)
        intrinsics = Ks[batch_ids]
        projected = (intrinsics[:, :2, :2] @ xy[..., None]).squeeze(-1) + intrinsics[:, :2, 2]
        errors = torch.linalg.vector_norm(projected - observations, dim=-1)
        # Sum tracks per view, average images in the minibatch. Delta is in
        # current training-image pixels; this is Huber on the 2D residual norm.
        loss = F.huber_loss(errors, torch.zeros_like(errors), reduction="sum", delta=huber_delta)
        count = valid.sum()
        mean_error = errors.detach().sum() / count.clamp_min(1)
        return loss / len(camtoworlds), mean_error, count
