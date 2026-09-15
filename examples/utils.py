import math
import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps


def so3_exp_map(omega: Tensor) -> Tensor:
    """Convert an axis-angle vector to a rotation matrix."""
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta = theta2.clamp_min(1e-12).sqrt()
    theta4 = theta2 * theta2
    small = theta2 < 1e-8
    A = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta4 / 120.0,
        torch.sin(theta) / theta,
    )
    B = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta4 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )

    x, y, z = omega.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*omega.shape[:-1], 3, 3)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype)
    eye = eye.expand(*omega.shape[:-1], 3, 3)
    return eye + A[..., None] * skew + B[..., None] * torch.matmul(skew, skew)


class CameraOptModule(torch.nn.Module):
    """Independent per-image rigid pose corrections with SO(3) rotations."""

    def __init__(self, n: int, reference_index: int | None = 0):
        super().__init__()
        if n <= 0:
            raise ValueError("Pose optimization requires at least one pose")
        if reference_index is not None and not 0 <= reference_index < n:
            raise ValueError("reference_index is outside the pose table")
        self.reference_index = reference_index
        self.trans = torch.nn.Embedding(n, 3, padding_idx=reference_index)
        self.rot = torch.nn.Embedding(n, 3, padding_idx=reference_index)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Apply a camera-local SE(3) correction to camera-to-world matrices."""
        return camtoworlds @ self._delta_transform(embed_ids)

    def zero_init(self):
        torch.nn.init.zeros_(self.trans.weight)
        torch.nn.init.zeros_(self.rot.weight)
        self._zero_reference()

    def random_init(self, std: float):
        torch.nn.init.normal_(self.trans.weight, std=std)
        torch.nn.init.normal_(self.rot.weight, std=std)
        self._zero_reference()

    def _zero_reference(self):
        if self.reference_index is not None:
            with torch.no_grad():
                self.trans.weight[self.reference_index].zero_()
                self.rot.weight[self.reference_index].zero_()

    def _delta_transform(self, pose_ids: Tensor) -> Tensor:
        dx = self.trans(pose_ids)
        drot = self.rot(pose_ids)
        rot = so3_exp_map(drot)
        transform = torch.eye(4, device=dx.device, dtype=dx.dtype)
        transform = transform.expand(*pose_ids.shape, 4, 4).clone()
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return transform

    @torch.no_grad()
    def metrics(self) -> dict[str, Tensor]:
        trans_norm = torch.linalg.vector_norm(self.trans.weight, dim=-1)
        rot_norm = torch.linalg.vector_norm(self.rot.weight, dim=-1)
        if self.reference_index is not None:
            mask = torch.ones_like(trans_norm, dtype=torch.bool)
            mask[self.reference_index] = False
            trans_norm, rot_norm = trans_norm[mask], rot_norm[mask]
        if trans_norm.numel() == 0:
            zero = self.trans.weight.new_zeros(())
            return {"translation_mean": zero, "translation_p95": zero,
                    "translation_max": zero, "rotation_mean_deg": zero,
                    "rotation_p95_deg": zero, "rotation_max_deg": zero}
        rotation_deg = rot_norm * (180.0 / math.pi)
        return {
            "translation_mean": trans_norm.mean(),
            "translation_p95": torch.quantile(trans_norm, 0.95),
            "translation_max": trans_norm.max(),
            "rotation_mean_deg": rotation_deg.mean(),
            "rotation_p95_deg": torch.quantile(rotation_deg, 0.95),
            "rotation_max_deg": rotation_deg.max(),
        }


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img
