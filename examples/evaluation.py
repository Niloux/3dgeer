from typing import Optional, Tuple

import torch
from torch import Tensor


def _valid_pixels(mask: Optional[Tensor], image: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    mask = mask.bool()
    if mask.ndim == image.ndim and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.shape != image.shape[:-1]:
        raise ValueError(
            f"Evaluation mask must have shape {image.shape[:-1]}, got {mask.shape}"
        )
    return mask


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
    """Compute legacy PSNR after black-padding ignored pixels in both images."""
    prediction, target, _ = prepare_evaluation_images(prediction, target, mask)
    mse = (prediction - target).square().mean()
    return -10.0 * torch.log10(mse)


def masked_ssim(
    metric, prediction: Tensor, target: Tensor, mask: Optional[Tensor]
) -> Tensor:
    """Compute legacy full-frame SSIM after black-padding ignored pixels."""
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


def masked_lpips(
    metric, prediction: Tensor, target: Tensor, mask: Optional[Tensor]
) -> Tensor:
    """Compute LPIPS after compositing ignored pixels to a common black background."""
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
