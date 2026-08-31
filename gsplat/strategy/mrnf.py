# MRNF behavior is adapted from LichtFeld Studio commit
# de972c89ec6bcd27406f892b966f180a7054f2cc (GPL-3.0-or-later).
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import _multinomial_sample, duplicate, remove, split


@dataclass
class MRNFStrategy(Strategy):
    """Error-attribution densification for Eval3D renderers.

    This ports the distortion-aware core of LichtFeld Studio's MRNF strategy:
    each pixel's reconstruction error is attributed to Gaussians using their
    actual ``alpha * transmittance`` contribution in the world-space backward
    pass. Edge guidance, far-field seeding, decay, and noise are intentionally
    outside this first implementation.
    """

    prune_opa: float = 1.0 / 255.0
    grow_error_threshold: float = 0.003
    grow_fraction: float = 0.07
    grow_scale3d: float = 0.01
    refine_start_iter: int = 0
    refine_stop_iter: int = 15_000
    refine_every: int = 200
    use_visibility_ratio: bool = True
    visibility_power: float = 0.75
    min_visibility: float = 0.05
    revised_opacity: bool = False
    max_gaussians: int = 1_000_000
    max_grow_per_refine: int = 0
    error_epsilon: float = 1e-6
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize lazily allocated per-Gaussian running statistics."""
        return {
            "visibility_sum": None,
            "error_max": None,
            "error_ratio_max": None,
            "scene_scale": float(scene_scale),
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
        assert self.grow_scale3d >= 0.0
        assert self.refine_every > 0
        assert self.refine_start_iter <= self.refine_stop_iter
        assert self.visibility_power >= 0.0
        assert self.min_visibility >= 0.0
        assert self.max_gaussians >= 0
        assert self.max_grow_per_refine >= 0
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
            n_duplicate, n_split = self._grow_gaussians(params, optimizers, state)
            for key in ("visibility_sum", "error_max", "error_ratio_max"):
                state[key].zero_()
            if self.verbose:
                print(
                    f"Step {step}: MRNF pruned {n_prune}, duplicated "
                    f"{n_duplicate}, and split {n_split} GSs. "
                    f"Now having {len(params['means'])} GSs."
                )

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
    ) -> Tuple[int, int]:
        error_max = state["error_max"]
        candidates = (error_max > self.grow_error_threshold) & (
            state["visibility_sum"] > 0.0
        )
        candidate_count = int(candidates.sum().item())
        n_grow = int(candidate_count * self.grow_fraction + 0.5)

        max_new = n_grow
        if self.max_gaussians > 0:
            max_new = min(
                max_new,
                max(0, self.max_gaussians - len(params["means"])),
            )
        if self.max_grow_per_refine > 0:
            max_new = min(max_new, self.max_grow_per_refine)
        n_grow = min(max_new, candidate_count)
        if n_grow <= 0:
            return 0, 0

        rank = (
            state["error_ratio_max"]
            if self.use_visibility_ratio
            else error_max
        )
        weights = torch.where(candidates, rank, torch.zeros_like(rank))
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1e10, neginf=0.0)
        if not bool((weights.sum() > 0.0).item()):
            weights = torch.where(candidates, error_max, torch.zeros_like(error_max))

        selected_ids = _multinomial_sample(weights, n_grow, replacement=False)
        selected = torch.zeros_like(candidates)
        selected[selected_ids] = True

        is_small = (
            torch.exp(params["scales"]).amax(dim=-1)
            <= self.grow_scale3d * state["scene_scale"]
        )
        duplicate_mask = selected & is_small
        split_mask = selected & ~is_small
        n_duplicate = int(duplicate_mask.sum().item())
        n_split = int(split_mask.sum().item())

        if n_duplicate > 0:
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=duplicate_mask,
            )
        split_mask = torch.cat(
            [
                split_mask,
                torch.zeros(
                    n_duplicate,
                    device=split_mask.device,
                    dtype=torch.bool,
                ),
            ]
        )
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=split_mask,
                revised_opacity=self.revised_opacity,
            )
        return n_duplicate, n_split
