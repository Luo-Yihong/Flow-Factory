# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conditional interval sampling in actual flow noise coordinates."""

from __future__ import annotations

import inspect
import math
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator

import torch


@contextmanager
def capture_generation_shift(scheduler: Any) -> Iterator[list[float]]:
    """Capture one rollout's effective shift while preserving the scheduler method.

    Args:
        scheduler: Primary scheduler used by the current generation call.

    Yields:
        A list containing the captured shift after a successful generation call.

    Raises:
        ValueError: No schedule was built, shifts differ within the rollout, or the
            schedule is not a supported pure flow shift.
    """
    original = scheduler.set_timesteps
    signature = inspect.signature(original)
    absent = object()
    instance_method = vars(scheduler).get("set_timesteps", absent)
    shifts: list[float] = []

    @wraps(original)
    def capture(*args: Any, **kwargs: Any) -> Any:
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        shift = _generation_shift(scheduler, arguments.arguments.get("mu"))
        result = original(*args, **kwargs)
        if shifts and shift != shifts[0]:
            raise ValueError("pre_shift_uniform requires one effective shift per rollout")
        if not shifts:
            shifts.append(shift)
        return result

    scheduler.set_timesteps = capture
    try:
        yield shifts
        if not shifts:
            raise ValueError("pre_shift_uniform did not capture a set_timesteps() call")
    finally:
        if instance_method is absent:
            del scheduler.set_timesteps
        else:
            scheduler.set_timesteps = instance_method


def _generation_shift(scheduler: Any, mu: float | None) -> float:
    """Resolve the effective rational flow shift from the actual generation call."""
    config = scheduler.config
    modifiers = [
        name
        for name in (
            "shift_terminal",
            "invert_sigmas",
            "use_karras_sigmas",
            "use_exponential_sigmas",
            "use_beta_sigmas",
        )
        if config.get(name, False)
    ]
    if "use_flow_sigmas" in config and not config.use_flow_sigmas:
        modifiers.append("use_flow_sigmas=False")
    if modifiers:
        raise ValueError(
            "pre_shift_uniform requires a pure flow-shift schedule; "
            f"unsupported schedule modifiers: {modifiers!r}"
        )
    if config.get("use_dynamic_shifting", False):
        kind = config.get("time_shift_type", "exponential")
        if mu is None or kind not in ("linear", "exponential"):
            raise ValueError(f"Cannot resolve generation shift: type={kind!r}, mu={mu!r}")
        shift = (
            (math.exp(mu) if mu <= math.log(sys.float_info.max) else float("inf"))
            if kind == "exponential"
            else mu
        )
    else:
        shift = getattr(scheduler, "shift", config.get("flow_shift"))
    if shift is None or not math.isfinite(shift) or shift <= 0:
        raise ValueError(f"Generation shift must be positive and finite, received {shift!r}")
    return float(shift)


def sample_interval_sigma(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    strategy: str,
    logit_mean: float,
    logit_std: float,
    shift: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw one independent sigma per interval using float64 probability arithmetic.

    Args:
        lower: Actual lower noise coordinates in [0, 1].
        upper: Actual upper noise coordinates in [0, 1], above lower.
        strategy: Conditional logit-normal or pre-shift uniform.
        logit_mean: Mean of the untruncated normal in logit space.
        logit_std: Positive standard deviation of that normal.
        shift: Per-sample effective shift captured during generation; uniform only.

    Returns:
        Float64 noise coordinates; callers protect representable output interiors.
    """
    lower, upper = lower.double(), upper.double()
    if not bool(((0 <= lower) & (lower < upper) & (upper <= 1)).all()):
        raise ValueError("TDM sigma intervals require 0 <= lower < upper <= 1")
    fraction = torch.rand(lower.shape, device=lower.device, dtype=torch.float64)
    eps = torch.finfo(torch.float64).eps
    fraction = fraction.clamp(eps, 1 - eps)
    if strategy == "pre_shift_uniform":
        if shift is None:
            raise ValueError("pre_shift_uniform requires the shift captured during generation")
        shift = shift.to(device=lower.device, dtype=torch.float64)
        if not bool((torch.isfinite(shift) & (shift > 0)).all()):
            raise ValueError("Generation shift must be positive and finite")
        u_lower = lower / (shift - (shift - 1) * lower)
        u_upper = upper / (shift - (shift - 1) * upper)
        uniform = u_lower + fraction * (u_upper - u_lower)
        return shift * uniform / (1 + (shift - 1) * uniform)
    if strategy != "truncated_logit_normal":
        raise ValueError(f"Unsupported TDM timestep sampling strategy: {strategy!r}")

    z_lower = (torch.logit(lower) - logit_mean) / logit_std
    z_upper = (torch.logit(upper) - logit_mean) / logit_std
    # Reflect right-tail intervals before evaluating the CDF, avoiding subtraction
    # of probabilities rounded to one. erfc also preserves far left-tail mass.
    reflect = z_lower > 0
    cdf_lower_z = torch.where(reflect, -z_upper, z_lower)
    cdf_upper_z = torch.where(reflect, -z_lower, z_upper)
    cdf_lower = 0.5 * torch.erfc(-cdf_lower_z / math.sqrt(2))
    cdf_upper = 0.5 * torch.erfc(-cdf_upper_z / math.sqrt(2))
    probability_lower = torch.nextafter(cdf_lower, cdf_upper)
    probability_upper = torch.nextafter(cdf_upper, cdf_lower)
    if not bool(((cdf_lower < cdf_upper) & (probability_lower < probability_upper)).all()):
        raise ValueError(
            "TDM truncated_logit_normal interval has no reliable float64 probability "
            f"interior: lower={lower.tolist()}, upper={upper.tolist()}, "
            f"tdm_logit_mean={logit_mean}, tdm_logit_std={logit_std}"
        )
    quantile = torch.where(reflect, 1 - fraction, fraction)
    probability = cdf_lower + quantile * (cdf_upper - cdf_lower)
    probability = torch.minimum(torch.maximum(probability, probability_lower), probability_upper)
    z = torch.special.ndtri(probability)
    z = torch.where(reflect, -z, z)
    return torch.sigmoid(logit_mean + logit_std * z)
