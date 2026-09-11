"""SH0 robust reconstruction with observation and Gaussian uncertainty.

Adapted from Hui et al., WACV 2026. This is not a numerical reproduction:
bounded KNN neighborhoods replace connected components; gradient statistics use
unweighted photometric probes; initial surface anisotropy is preserved. All
weights are detached statistics, so no second-order gradients are required.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.strategy.mrnf import _ssim_cs_error_map


@dataclass
class UncertaintyConfig:
    enabled: bool = False
    warmup_steps: int = 1500
    ramp_steps: int = 1500
    refresh_every: int = 500
    probe_every: int = 50
    phase_steps: int = 2000
    neighbors: int = 8
    min_views: int = 3
    min_observations: int = 8
    ema_decay: float = 0.9
    min_weight: float = 0.15
    min_survival: float = 0.1
    fusion_weight: float = 0.5
    opacity_strength: float = 2.0
    temperature_start: float = 2.0
    temperature_end: float = 0.5
    outlier_threshold: float = 1.5

    def validate(self):
        if self.warmup_steps < 0 or self.ramp_steps < 1:
            raise ValueError("Uncertainty warmup must be >= 0 and ramp must be > 0")
        for name in ("refresh_every", "probe_every", "phase_steps", "neighbors",
                     "min_observations"):
            if getattr(self, name) < 1:
                raise ValueError(f"uncertainty.{name} must be positive")
        if self.min_views < 2:
            raise ValueError("uncertainty.min_views must be at least 2")
        for name in ("min_weight", "min_survival"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"uncertainty.{name} must be in (0, 1]")
        if not 0.0 <= self.ema_decay < 1.0 or not 0 <= self.fusion_weight <= 1:
            raise ValueError("Invalid uncertainty EMA decay or fusion weight")
        for name in ("opacity_strength", "temperature_start", "temperature_end",
                     "outlier_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"uncertainty.{name} must be finite and positive")


def _positive_z(values: Tensor, valid: Tensor) -> Tensor:
    """Robust phase normalization; no evidence is neutral, not an outlier."""
    selected = values[valid]
    if selected.numel() < 2:
        return torch.zeros_like(values)
    q25, q50, q75 = torch.quantile(selected, selected.new_tensor([.25, .5, .75]))
    scale = torch.maximum((q75 - q25) / 1.349, selected.std(unbiased=False) * .1)
    return torch.where(valid, ((values - q50) / scale.clamp_min(1e-4)).clamp(0, 6), 0)


def _percentile_score(values: Tensor) -> Tensor:
    if values.numel() < 2:
        return torch.zeros_like(values)
    lo, hi = torch.quantile(values, values.new_tensor([.05, .95]))
    return ((values - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


class ReconstructionUncertainty:
    """Detached statistics; per-splat tensors live in MRNF's mutation state."""

    def __init__(self, cfg: UncertaintyConfig, state: Dict[str, Any],
                 splats, num_images: int, max_steps: int):
        cfg.validate()
        self.cfg, self.state, self.max_steps = cfg, state, max_steps
        self.step = 0
        self.image_errors = torch.zeros(num_images, 32, 32, device=splats["means"].device)
        self.image_seen = torch.zeros(num_images, dtype=torch.bool, device=splats["means"].device)
        self.error_scale = splats["means"].new_tensor(.05)
        self.last_metrics = {}
        self.initialize(splats)

    def initialize(self, splats):
        n, device = len(splats["means"]), splats["means"].device
        for name in ("age", "grad_mean", "grad_square", "grad_count", "utilization",
                     "shape_base", "aleatoric", "epistemic", "survival", "confidence",
                     "observation_confidence"):
            key = "uncertainty_" + name
            if key not in self.state or self.state[key].shape != (n,):
                fill = 1.0 if name in ("survival", "confidence", "observation_confidence") else 0.0
                self.state[key] = torch.full((n,), fill, device=device)
        key = "uncertainty_view_ids"
        if key not in self.state or self.state[key].shape != (n, self.cfg.min_views):
            self.state[key] = torch.full((n, self.cfg.min_views), -1, device=device,
                                        dtype=torch.int32)

    def get(self, name):
        return self.state["uncertainty_" + name]

    @property
    def strength(self):
        return min(1.0, max(0.0, (self.step - self.cfg.warmup_steps) / self.cfg.ramp_steps))

    @property
    def temperature(self):
        t = min(1.0, max(0.0, (self.step - self.cfg.warmup_steps) /
                         max(1, self.max_steps - self.cfg.warmup_steps)))
        return self.cfg.temperature_start * (self.cfg.temperature_end /
                                              self.cfg.temperature_start) ** t

    def _confidence(self, score):
        threshold = self.cfg.outlier_threshold / self.temperature
        normalizer = 1.0 / (1.0 + math.exp(-threshold))
        confidence = torch.sigmoid((self.cfg.outlier_threshold - score) / self.temperature)
        confidence = self.cfg.min_weight + (1 - self.cfg.min_weight) * confidence / normalizer
        return 1 - self.strength * (1 - confidence.clamp(max=1))

    @torch.no_grad()
    def prepare(self, step, splats):
        self.step = step
        age = self.get("age")
        shape = splats["scales"].amax(-1) - splats["scales"].amin(-1)
        self.get("shape_base").copy_(torch.where(age == 0, shape, self.get("shape_base")))
        if step and step % self.cfg.phase_steps == 0:
            for name in ("grad_mean", "grad_square", "grad_count"):
                self.get(name).zero_()
            self.get("aleatoric").zero_()
        if step >= self.cfg.warmup_steps and step % self.cfg.refresh_every == 0:
            self._refresh(splats, shape)
        fused = (self.cfg.fusion_weight * self.get("aleatoric") +
                 (1 - self.cfg.fusion_weight) * self.get("epistemic"))
        observation_confidence = 1 - self.strength * (1 - self.get("observation_confidence"))
        self.get("confidence").copy_(torch.minimum(self._confidence(fused), observation_confidence))
        survival = torch.exp(-self.cfg.opacity_strength * self.get("epistemic"))
        self.get("survival").copy_(1 - self.strength *
                                   (1 - survival.clamp_min(self.cfg.min_survival)))

    @torch.no_grad()
    def _refresh(self, splats, shape):
        from scipy.spatial import cKDTree

        count = self.get("grad_count")
        mean = self.get("grad_mean")
        std = (self.get("grad_square") - mean.square()).clamp_min(0).sqrt()
        measured = count >= 2
        gradient_score = _positive_z(mean + std, measured)
        n = len(mean)
        if n == 0:
            return
        # Bounded neighborhoods avoid a single connected component spanning an
        # entire wall. Query in chunks: never allocate an N x N distance matrix.
        xyz = splats["means"].detach().cpu().double().numpy()
        tree = cKDTree(xyz)
        k = min(self.cfg.neighbors + 1, n)
        distances = torch.zeros_like(mean)
        spatial_score = torch.zeros_like(mean)
        for start in range(0, n, 32768):
            end = min(start + 32768, n)
            if k == 1:
                spatial_score[start:end] = gradient_score[start:end]
                continue
            d, neighbors = tree.query(xyz[start:end], k=k, workers=1)
            ids = torch.as_tensor(neighbors, device=mean.device, dtype=torch.long)
            d = torch.as_tensor(d, device=mean.device, dtype=mean.dtype)
            valid = measured[ids]
            spatial_score[start:end] = (gradient_score[ids] * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            distances[start:end] = d[:, 1:].mean(-1)
        # Retain individual outliers rather than smoothing every signal away.
        self.get("aleatoric").copy_(torch.where(measured, torch.maximum(gradient_score, spatial_score), 0))
        views = (self.get("view_ids") >= 0).sum(-1)
        deficit = (1 - (views - 1).clamp_min(0) / (self.cfg.min_views - 1)).clamp(0, 1)
        low_util = 1 - _percentile_score(torch.log1p(self.get("utilization")))
        # Thin LiDAR surfels are intentional: penalize *growth* in anisotropy.
        anisotropy = _percentile_score((shape - self.get("shape_base")).clamp_min(0))
        isolation = _percentile_score(distances)
        epistemic = deficit * (.5 + (low_util + anisotropy + isolation) / 6)
        mature = self.get("age") >= self.cfg.min_observations
        self.get("epistemic").copy_(torch.where(mature, epistemic, 0))

    def features(self):
        return torch.stack((self.get("aleatoric"), self.get("epistemic")), dim=-1)

    def effective_opacity(self, logits):
        return torch.sigmoid(logits) * self.get("survival")

    def export_opacity(self, logits):
        return torch.logit(self.effective_opacity(logits), eps=1e-6)

    @torch.no_grad()
    def pixel_weights(self, prediction, target, mask, projected, image_id):
        valid = torch.ones_like(target[..., 0], dtype=torch.bool) if mask is None else mask
        error = (prediction - target).abs().mean(-1) + .2 * _ssim_cs_error_map(prediction, target)
        error = torch.nan_to_num(error, nan=6.0, posinf=6.0, neginf=6.0)
        # An 11x11 footprint matches structural supervision. Exclude invalid
        # samples before pooling; they must not lower the local error estimate.
        numerator = F.avg_pool2d((error * valid).unsqueeze(1), 11, 1, 5)
        fraction = F.avg_pool2d(valid.float().unsqueeze(1), 11, 1, 5)
        local = numerator / fraction.clamp_min(1e-6)
        grid_num = F.adaptive_avg_pool2d((error * valid).unsqueeze(1), (32, 32))[0, 0]
        grid_valid = F.adaptive_avg_pool2d(valid.float().unsqueeze(1), (32, 32))[0, 0]
        grid = grid_num / grid_valid.clamp_min(1e-6)
        seen = self.image_seen[image_id]
        history = F.interpolate(self.image_errors[image_id][None, None],
                                size=error.shape[-2:], mode="bilinear", align_corners=False)
        local = torch.where(seen, .75 * local + .25 * history, local).squeeze(1)
        values = local[valid]
        if values.numel():
            baseline = torch.quantile(values, .5).clamp_min(.01)
            reference = torch.minimum(baseline, self.error_scale.clamp_min(.01) * 2)
            local_score = ((local - reference) / reference).clamp(0, 6)
            self.error_scale.lerp_(baseline, 1 - self.cfg.ema_decay)
        else:
            local_score = torch.zeros_like(error)
        updated = torch.where(seen, self.image_errors[image_id] * self.cfg.ema_decay +
                              grid * (1 - self.cfg.ema_decay), grid)
        self.image_errors[image_id].copy_(torch.where(grid_valid > 0, updated,
                                                    self.image_errors[image_id]))
        self.image_seen[image_id] = True
        ua, ue = projected.unbind(-1)
        score = self.cfg.fusion_weight * torch.maximum(local_score, ua) + (1 - self.cfg.fusion_weight) * ue
        weight = self._confidence(score).masked_fill(~valid, 0)
        denominator = valid.sum().clamp_min(1)
        self.last_metrics = {
            "uncertainty_weight_mean": weight.sum() / denominator,
            "uncertainty_low_weight_fraction": ((weight < .5) & valid).sum() / denominator,
            "uncertainty_strength": self.strength,
            "uncertainty_temperature": self.temperature,
            "uncertainty_aleatoric_mean": self.get("aleatoric").mean(),
            "uncertainty_epistemic_mean": self.get("epistemic").mean(),
            "uncertainty_survival_mean": self.get("survival").mean(),
        }
        return weight

    @torch.no_grad()
    def observe_visibility(self, info, image_id):
        utilization = info["densification_info"].reshape(-1, 2, len(self.get("age")))[:, 0].sum(0)
        visible = utilization > 1e-4
        self.get("age").add_(visible)
        old = self.get("utilization")
        sample = utilization / max(1, info["width"] * info["height"])
        old.copy_(torch.where(visible, old * self.cfg.ema_decay + sample * (1 - self.cfg.ema_decay), old))
        views = self.get("view_ids")
        unseen = visible & ~(views == image_id).any(-1) & (views < 0).any(-1)
        ids = torch.where(unseen)[0]
        slot = (views[ids] >= 0).sum(-1)
        views[ids, slot] = image_id

    @torch.no_grad()
    def observe_confidence(self, info):
        stats = info["densification_info"].reshape(-1, 2, len(self.get("age"))).sum(0)
        visible = stats[0] > 1e-4
        confidence = (stats[1] / stats[0].clamp_min(1e-6)).clamp(0, 1)
        old = self.get("observation_confidence")
        old.copy_(torch.where(visible, old * self.cfg.ema_decay +
                             confidence * (1 - self.cfg.ema_decay), old))

    @torch.no_grad()
    def observe_gradients(self, grads, visible):
        if not bool(visible.any()):
            return
        total = torch.zeros_like(self.get("grad_mean"))
        for grad in grads:
            if grad is None or grad.numel() == 0:
                continue
            energy = grad.detach().reshape(len(total), -1).square().sum(-1)
            normalization = energy[visible].mean().clamp_min(1e-20)
            total.add_(energy / normalization)
        magnitude = torch.log1p(total.sqrt())
        valid = visible & torch.isfinite(magnitude)
        decay = torch.where(self.get("grad_count") == 0, 0., self.cfg.ema_decay)
        for name, sample in (("grad_mean", magnitude), ("grad_square", magnitude.square())):
            old = self.get(name)
            old.copy_(torch.where(valid, decay * old + (1 - decay) * sample, old))
        self.get("grad_count").add_(valid)

    def state_dict(self):
        return {"version": 1, "config": asdict(self.cfg), "step": self.step,
                "gaussians": {k: v for k, v in self.state.items() if k.startswith("uncertainty_")},
                "image_errors": self.image_errors, "image_seen": self.image_seen,
                "error_scale": self.error_scale}

    def load_state_dict(self, saved, splats):
        if saved.get("version") != 1:
            raise ValueError("Unsupported uncertainty checkpoint version")
        tensors = saved["gaussians"]
        n = len(splats["means"])
        if set(tensors) != {k for k in self.state if k.startswith("uncertainty_")}:
            raise ValueError("Uncertainty checkpoint is missing Gaussian statistics")
        for key, value in tensors.items():
            expected = (n, self.cfg.min_views) if key.endswith("view_ids") else (n,)
            if tuple(value.shape) != expected:
                raise ValueError(f"Invalid uncertainty checkpoint shape for {key}")
            self.state[key] = value.to(splats["means"].device)
        if saved["image_errors"].shape != self.image_errors.shape:
            raise ValueError("Uncertainty checkpoint training image count does not match")
        self.image_errors.copy_(saved["image_errors"])
        self.image_seen.copy_(saved["image_seen"])
        self.error_scale.copy_(saved["error_scale"])
        self.step = int(saved["step"])
