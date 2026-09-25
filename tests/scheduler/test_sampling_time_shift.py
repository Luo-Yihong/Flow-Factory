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

"""Generation is the sole owner of effective static and dynamic flow shifts."""

import math

import numpy as np
import pytest
import torch

from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from flow_factory.scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    MiniMaxH3SDEScheduler,
    UniPCMultistepSDEScheduler,
)


@pytest.mark.parametrize("kind", ["static", "exponential", "linear"])
@pytest.mark.parametrize("family", ["euler", "unipc"])
def test_sampling_shift_matches_upstream_schedule_bitwise(kind, family):
    dynamic = kind != "static"
    config = dict(use_dynamic_shifting=dynamic, time_shift_type=kind if dynamic else "exponential")
    if family == "euler":
        cls, upstream = FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteScheduler
        config["shift"] = 3
    else:
        cls, upstream = UniPCMultistepSDEScheduler, UniPCMultistepScheduler
        config.update(use_flow_sigmas=True, flow_shift=3)
    scheduler, reference = cls(**config), upstream(**config)
    with pytest.raises(ValueError, match="set_timesteps"):
        _ = scheduler.sampling_time_shift
    for mu in (0.4, 0.9):
        kwargs = dict(
            sigmas=np.array([1.0, 0.75, 0.5, 0.25]),
            num_inference_steps=4,
            mu=mu if dynamic else None,
        )
        scheduler.set_timesteps(**kwargs)
        reference.set_timesteps(**kwargs)
        assert torch.equal(scheduler.sigmas, reference.sigmas)
        assert torch.equal(scheduler.timesteps, reference.timesteps)
        gamma = math.exp(mu) if kind == "exponential" else mu if kind == "linear" else 3
        assert scheduler.sampling_time_shift == pytest.approx(gamma)
        raw = np.array([1.0, 0.75, 0.5, 0.25])
        expected = gamma * raw / (1 + (gamma - 1) * raw)
        np.testing.assert_allclose(scheduler.sigmas[:-1].numpy(), expected, atol=2e-6, rtol=0)


def test_h3_captures_static_shift_before_mutation():
    scheduler = MiniMaxH3SDEScheduler(shift=12)
    scheduler.set_timesteps(4)
    scheduler.set_shift(8)
    assert scheduler.sampling_time_shift == 12
    scheduler.set_timesteps(4)
    assert scheduler.sampling_time_shift == 8


@pytest.mark.parametrize(
    "config", [{"shift_terminal": 0.02}, {"invert_sigmas": True}, {"use_karras_sigmas": True}]
)
def test_non_shift_schedule_rejected_without_changing_generation(config):
    scheduler = FlowMatchEulerDiscreteSDEScheduler(**config)
    scheduler.set_timesteps(sigmas=[1, 0.75, 0.5, 0.25])
    with pytest.raises(ValueError, match="pure flow-shift schedule"):
        _ = scheduler.sampling_time_shift
