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

import math

import torch


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
