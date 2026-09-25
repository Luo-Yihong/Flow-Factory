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

"""Expose the effective flow shift from the most recently generated schedule."""

import math
import sys


class FlowSamplingTimeShiftMixin:
    """Retain schedule-time shift parameters without changing schedule arithmetic."""

    def _record_sampling_time_shift(self, *, static_shift: float, mu: float | None) -> None:
        self._sampling_shift_parameters = (
            bool(self.config.get("use_dynamic_shifting", False)),
            self.config.get("time_shift_type", "exponential"),
            static_shift,
            mu,
        )
        self._sampling_shift_modifiers = tuple(
            name
            for name in (
                "shift_terminal",
                "invert_sigmas",
                "use_karras_sigmas",
                "use_exponential_sigmas",
                "use_beta_sigmas",
            )
            if self.config.get(name, False)
        )
        if "use_flow_sigmas" in self.config and not self.config.use_flow_sigmas:
            self._sampling_shift_modifiers += ("use_flow_sigmas=False",)

    @property
    def sampling_time_shift(self) -> float:
        """Return the effective rational shift used by the last generated schedule.

        Returns:
            Positive gamma in S(u) = gamma*u / (1 + (gamma-1)*u).

        Raises:
            ValueError: The schedule has not been built or is not a pure flow shift.
        """
        parameters = getattr(self, "_sampling_shift_parameters", None)
        if parameters is None:
            raise ValueError("pre_shift_uniform requires set_timesteps() before shift capture")
        if self._sampling_shift_modifiers:
            raise ValueError(
                "pre_shift_uniform requires a pure flow-shift schedule; unsupported "
                f"schedule modifiers: {self._sampling_shift_modifiers!r}"
            )
        dynamic, kind, static_shift, mu = parameters
        if dynamic:
            if mu is None or kind not in ("linear", "exponential"):
                raise ValueError(f"Cannot resolve generation shift: type={kind!r}, mu={mu!r}")
            if kind == "exponential":
                shift = math.exp(mu) if mu <= math.log(sys.float_info.max) else float("inf")
            else:
                shift = mu
        else:
            shift = static_shift
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"Generation shift must be positive and finite, received {shift!r}")
        return float(shift)
