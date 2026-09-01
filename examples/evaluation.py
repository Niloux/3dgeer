from collections.abc import Sequence
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure


def _valid_pixels(mask: Optional[Tensor], image: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    mask = mask.to(device=image.device, dtype=torch.bool)
    if mask.ndim == image.ndim and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.shape != image.shape[:-1]:
        raise ValueError(
            f"Evaluation mask must have shape {image.shape[:-1]}, got {mask.shape}"
        )
    return mask


def _validate_image_pair(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target must have the same shape, got "
            f"{prediction.shape} and {target.shape}"
        )
    if prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError(
            "Evaluation images must have shape [B, H, W, 3], got "
            f"{prediction.shape}"
        )


def _per_image_spatial_mean(
    values: Tensor,
    valid: Optional[Tensor],
    metric_name: str,
) -> Tensor:
    """Average NCHW values over valid pixels, returning one value per image."""
    if valid is None:
        return values.flatten(1).mean(dim=1)
    if valid.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError(
            f"{metric_name} mask has shape {valid.shape}, but values have shape "
            f"{values.shape}"
        )
    weights = valid[:, None].to(dtype=values.dtype)
    denominator = weights.sum(dim=(1, 2, 3)) * values.shape[1]
    if bool((denominator == 0).any().item()):
        raise ValueError(f"{metric_name} has no strictly valid samples")
    return (values * weights).sum(dim=(1, 2, 3)) / denominator


def prepare_evaluation_images(
    prediction: Tensor,
    target: Tensor,
    mask: Optional[Tensor],
    background: float = 0.0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Composite ignored pixels to the same value and return the valid selector."""
    valid = _valid_pixels(mask, prediction)
    if valid is None:
        return prediction, target, None
    fill = prediction.new_full((1,), background)
    valid_rgb = valid[..., None]
    return (
        torch.where(valid_rgb, prediction, fill),
        torch.where(valid_rgb, target, fill),
        valid,
    )


def masked_psnr(prediction: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Compute PSNR from valid RGB samples only."""
    _validate_image_pair(prediction, target)
    valid = _valid_pixels(mask, prediction)
    squared_error = (prediction - target).square().permute(0, 3, 1, 2)
    mse = _per_image_spatial_mean(squared_error, valid, "PSNR")
    return (-10.0 * torch.log10(mse)).mean()


def _pair(value) -> Tuple[float, float]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError(f"Expected a 2D value, got {value}")
        return float(value[0]), float(value[1])
    scalar = float(value)
    return scalar, scalar


def _ssim_window_size(metric) -> Tuple[int, int]:
    if metric.gaussian_kernel:
        sigma_h, sigma_w = _pair(metric.sigma)
        return (
            int(3.5 * sigma_h + 0.5) * 2 + 1,
            int(3.5 * sigma_w + 0.5) * 2 + 1,
        )
    kernel_h, kernel_w = _pair(metric.kernel_size)
    return int(kernel_h), int(kernel_w)


def _fully_valid_ssim_windows(valid: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    """Return SSIM centers whose complete reflected window is valid."""
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = (kernel_h - 1) // 2, (kernel_w - 1) // 2
    valid_f = valid[:, None].to(dtype=torch.float32)
    valid_f = F.pad(valid_f, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    coverage = F.avg_pool2d(valid_f, kernel_size, stride=1)
    return coverage[:, 0] >= 1.0 - 1e-6


def masked_ssim(
    metric, prediction: Tensor, target: Tensor, mask: Optional[Tensor]
) -> Tensor:
    """Compute SSIM from windows whose entire support lies in the valid domain."""
    _validate_image_pair(prediction, target)
    prediction, target, valid = prepare_evaluation_images(
        prediction, target, mask
    )
    _, ssim_map = structural_similarity_index_measure(
        prediction.permute(0, 3, 1, 2),
        target.permute(0, 3, 1, 2),
        gaussian_kernel=metric.gaussian_kernel,
        sigma=metric.sigma,
        kernel_size=metric.kernel_size,
        reduction="none",
        data_range=metric.data_range,
        k1=metric.k1,
        k2=metric.k2,
        return_full_image=True,
    )
    valid_windows = (
        _fully_valid_ssim_windows(valid, _ssim_window_size(metric))
        if valid is not None
        else None
    )
    return _per_image_spatial_mean(ssim_map, valid_windows, "SSIM").mean()


def masked_lpips(
    metric, prediction: Tensor, target: Tensor, mask: Optional[Tensor]
) -> Tensor:
    """Compute standard LPIPS after compositing ignored pixels to black."""
    _validate_image_pair(prediction, target)
    prediction, target, _ = prepare_evaluation_images(prediction, target, mask)
    metric.reset()
    try:
        value = metric(
            prediction.permute(0, 3, 1, 2),
            target.permute(0, 3, 1, 2),
        )
        return value.detach().clone()
    finally:
        metric.reset()
