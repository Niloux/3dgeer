# MRNF behavior is adapted from LichtFeld Studio commit
# de972c89ec6bcd27406f892b966f180a7054f2cc (GPL-3.0-or-later).
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

import math
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat

from .base import Strategy
from .ops import _multinomial_sample, _update_param_with_optimizer, remove


@torch.no_grad()
def _long_axis_split_gaussians(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Any],
    selected_ids: Tensor,
) -> None:
    """Split selected Gaussians with LichtFeld's in-place long-axis rule."""
    selected_ids = selected_ids.to(dtype=torch.long)
    selected_means = params["means"][selected_ids]
    selected_log_scales = params["scales"][selected_ids]
    selected_quats = params["quats"][selected_ids]

    longest_axes = selected_log_scales.argmax(dim=-1)
    rotmats = normalized_quat_to_rotmat(selected_quats)
    world_axes = torch.gather(
        rotmats,
        dim=2,
        index=longest_axes[:, None, None].expand(-1, 3, 1),
    ).squeeze(2)
    longest_scales = torch.gather(
        selected_log_scales, dim=1, index=longest_axes[:, None]
    ).exp()
    offsets = world_axes * (0.5 * longest_scales)

    child_means = selected_means - offsets
    parent_means = selected_means + offsets
    split_log_scales = selected_log_scales + math.log(0.85)
    split_log_scales.scatter_(
        1,
        longest_axes[:, None],
        torch.gather(
            selected_log_scales, dim=1, index=longest_axes[:, None]
        )
        + math.log(0.5),
    )
    split_opacities = torch.logit(
        torch.sigmoid(params["opacities"][selected_ids]) * 0.6,
        eps=1e-7,
    )

    def param_fn(name: str, parameter: Tensor) -> Tensor:
        if name == "means":
            parameter[selected_ids] = parent_means
            children = child_means
        elif name == "scales":
            parameter[selected_ids] = split_log_scales
            children = split_log_scales
        elif name == "opacities":
            parameter[selected_ids] = split_opacities
            children = split_opacities
        else:
            children = parameter[selected_ids]
        return torch.nn.Parameter(
            torch.cat((parameter, children), dim=0),
            requires_grad=parameter.requires_grad,
        )

    def optimizer_fn(key: str, value: Tensor) -> Tensor:
        del key
        value[selected_ids] = 0
        return torch.cat((value, torch.zeros_like(value[selected_ids])), dim=0)

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for key, value in state.items():
        if isinstance(value, Tensor):
            value[selected_ids] = 0
            state[key] = torch.cat(
                (value, torch.zeros_like(value[selected_ids])), dim=0
            )


@dataclass
class MRNFStrategy(Strategy):
    """Error-attribution densification for Eval3D renderers.

    This ports the distortion-aware core of LichtFeld Studio's MRNF strategy:
    each pixel's reconstruction error is attributed to Gaussians using their
    actual ``alpha * transmittance`` contribution in the world-space backward
    pass, and selected Gaussians use its in-place long-axis split rule. Opacity
    and scale decay follow LichtFeld's activated-space schedule. Edge guidance,
    far-field seeding, and noise remain outside this port.
    """

    prune_opa: float = 1.0 / 255.0
    grow_error_threshold: float = 0.003
    grow_fraction: float = 0.07
    refine_start_iter: int = 0
    grow_until_iter: int = 15_000
    refine_stop_iter: int = 28_500
    refine_every: int = 200
    use_visibility_ratio: bool = False
    visibility_power: float = 0.75
    min_visibility: float = 0.05
    max_gaussians: int = 1_000_000
    bounds_percentile: float = 0.8
    opacity_decay: float = 0.004
    scale_decay: float = 0.002
    error_epsilon: float = 1e-6
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize lazily allocated per-Gaussian running statistics."""
        del scene_scale
        return {
            "visibility_sum": None,
            "error_max": None,
            "error_ratio_max": None,
        }

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> None:
        super().check_sanity(params, optimizers)
        for key in ("means", "scales", "quats", "opacities"):
            assert key in params, f"{key} is required in params but missing."
        assert 0.0 <= self.grow_fraction <= 1.0
        assert self.grow_error_threshold >= 0.0
        assert self.refine_every > 0
        assert self.refine_start_iter <= self.grow_until_iter
        assert self.grow_until_iter <= self.refine_stop_iter
        assert self.visibility_power >= 0.0
        assert self.min_visibility >= 0.0
        assert self.max_gaussians >= 0
        assert 0.0 <= self.bounds_percentile <= 1.0
        assert self.opacity_decay >= 0.0
        assert 0.0 <= self.scale_decay < 1.0
        assert self.error_epsilon > 0.0

    def should_collect(self, step: int) -> bool:
        """Whether this step needs renderer-side MRNF attribution buffers."""
        return step < self.refine_stop_iter

    def _ensure_running_stats(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
    ) -> None:
        means = params["means"]
        n_gaussians = len(means)
        for key in ("visibility_sum", "error_max", "error_ratio_max"):
            value = state.get(key)
            if (
                not isinstance(value, Tensor)
                or value.shape != (n_gaussians,)
                or value.device != means.device
            ):
                state[key] = torch.zeros(
                    n_gaussians, device=means.device, dtype=torch.float32
                )

    @torch.no_grad()
    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        rendered: Tensor,
        target: Tensor,
        mask: Union[Tensor, None] = None,
    ) -> None:
        """Write a detached, valid-pixel-normalized RGB error map for backward."""
        del params, optimizers, state
        if not self.should_collect(step):
            return
        assert "densification_error_map" in info, (
            "MRNFStrategy requires rasterization(..., "
            "calc_densification_info=True)."
        )
        error_map = info["densification_error_map"]
        assert rendered.shape == target.shape, (rendered.shape, target.shape)
        assert rendered.shape[-1] >= 3, rendered.shape

        error = (rendered[..., :3] - target[..., :3]).abs().mean(dim=-1).detach()
        assert error.shape == error_map.shape, (error.shape, error_map.shape)

        if mask is None:
            valid = torch.ones_like(error, dtype=torch.bool)
        else:
            valid = mask.to(device=error.device, dtype=torch.bool)
            assert valid.shape == error.shape, (valid.shape, error.shape)
        valid_f = valid.to(dtype=error.dtype)
        valid_mean = (error * valid_f).sum() / valid_f.sum().clamp_min(1.0)
        normalized = torch.where(
            valid,
            error / valid_mean.clamp_min(self.error_epsilon),
            torch.zeros_like(error),
        )
        error_map.copy_(normalized.to(dtype=error_map.dtype))

    @torch.no_grad()
    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        max_steps: int,
    ) -> None:
        """Fold renderer attribution into the current refinement window."""
        if not self.should_collect(step):
            return
        assert "densification_info" in info, (
            "MRNFStrategy requires Eval3D densification statistics."
        )
        densification_info = info["densification_info"]
        n_gaussians = len(params["means"])
        assert densification_info.shape[-2:] == (2, n_gaussians), (
            densification_info.shape
        )

        self._ensure_running_stats(params, state)
        stats = densification_info.reshape(-1, 2, n_gaussians)
        visibility = stats[:, 0, :]
        weighted_error = stats[:, 1, :]

        state["visibility_sum"].add_(visibility.sum(dim=0))
        state["error_max"].copy_(
            torch.maximum(state["error_max"], weighted_error.amax(dim=0))
        )
        ratio = torch.where(
            visibility >= self.min_visibility,
            weighted_error / visibility.clamp_min(self.error_epsilon).pow(
                self.visibility_power
            ),
            torch.zeros_like(weighted_error),
        )
        state["error_ratio_max"].copy_(
            torch.maximum(state["error_ratio_max"], ratio.amax(dim=0))
        )
        densification_info.zero_()

        if step > self.refine_start_iter and step % self.refine_every == 0:
            n_prune = self._prune_gaussians(params, optimizers, state)
            n_split = self._grow_gaussians(
                params,
                optimizers,
                state,
                pruned_count=n_prune,
                allow_growth=step < self.grow_until_iter,
            )
            self._apply_decay(params, step=step, max_steps=max_steps)
            for key in ("visibility_sum", "error_max", "error_ratio_max"):
                state[key].zero_()
            if self.verbose:
                print(
                    f"Step {step}: MRNF pruned {n_prune} and split "
                    f"{n_split} GSs. "
                    f"Now having {len(params['means'])} GSs."
                )

    @torch.no_grad()
    def _apply_decay(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        step: int,
        max_steps: int,
    ) -> None:
        """Apply LichtFeld's linearly annealed opacity and scale decay."""
        assert max_steps > 0
        if self.opacity_decay == 0.0 and self.scale_decay == 0.0:
            return

        train_t = min(max(float(step) / float(max_steps), 0.0), 1.0)
        shrink_t = 1.0 - train_t
        if self.opacity_decay > 0.0:
            opacity = torch.sigmoid(params["opacities"])
            opacity.sub_(self.opacity_decay * shrink_t)
            opacity.clamp_(1e-12, 1.0 - 1e-12)
            params["opacities"].copy_(torch.logit(opacity))
        if self.scale_decay > 0.0:
            decay_factor = 1.0 - self.scale_decay * shrink_t
            params["scales"].add_(math.log(decay_factor))
            params["scales"].clamp_min_(math.log(1e-12))

    @torch.no_grad()
    def _prune_gaussians(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> int:
        prune_mask = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        n_prune = int(prune_mask.sum().item())
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=prune_mask)
        return n_prune

    @torch.no_grad()
    def _grow_gaussians(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        pruned_count: int,
        allow_growth: bool,
    ) -> int:
        error_max = state["error_max"]
        candidates = (error_max > self.grow_error_threshold) & (
            state["visibility_sum"] > 0.0
        )
        candidate_count = int(candidates.sum().item())
        budget = candidate_count + pruned_count
        if self.max_gaussians > 0:
            budget = min(
                budget,
                max(0, self.max_gaussians - len(params["means"])),
            )
        if budget <= 0:
            return 0

        replacement_weights = torch.where(
            state["visibility_sum"] > 0.0,
            torch.sigmoid(params["opacities"].flatten()),
            torch.zeros_like(error_max),
        )
        replacement_weights = torch.nan_to_num(
            replacement_weights, nan=0.0, posinf=1.0, neginf=0.0
        )
        selectable_replacements = int((replacement_weights > 0.0).sum().item())
        n_replace = min(pruned_count, budget, selectable_replacements)
        if n_replace > 0:
            replacement_ids = _multinomial_sample(
                replacement_weights,
                n_replace,
                replacement=False,
            )
        else:
            replacement_ids = torch.empty(
                0, device=error_max.device, dtype=torch.long
            )

        n_grow = 0
        if allow_growth:
            desired_total = int(candidate_count * self.grow_fraction + 0.5)
            n_grow = min(
                max(0, desired_total - n_replace),
                max(0, budget - n_replace),
            )

        if n_grow > 0:
            rank = (
                state["error_ratio_max"]
                if self.use_visibility_ratio
                else error_max
            )
            growth_weights = torch.where(candidates, rank, torch.zeros_like(rank))
            growth_weights[replacement_ids] = 0
            growth_weights = torch.nan_to_num(
                growth_weights, nan=0.0, posinf=1e10, neginf=0.0
            )
            if not bool((growth_weights.sum() > 0.0).item()):
                growth_weights = torch.where(
                    candidates, error_max, torch.zeros_like(error_max)
                )
                growth_weights[replacement_ids] = 0
            selectable_growth = int((growth_weights > 0.0).sum().item())
            n_grow = min(n_grow, selectable_growth)
            if n_grow > 0:
                growth_ids = _multinomial_sample(
                    growth_weights,
                    n_grow,
                    replacement=False,
                )
            else:
                growth_ids = replacement_ids.new_empty(0)
        else:
            growth_ids = replacement_ids.new_empty(0)

        selected_ids = torch.cat((replacement_ids, growth_ids))
        n_split = len(selected_ids)
        if n_split == 0:
            return 0
        _long_axis_split_gaussians(
            params=params,
            optimizers=optimizers,
            state=state,
            selected_ids=selected_ids,
        )
        return n_split
