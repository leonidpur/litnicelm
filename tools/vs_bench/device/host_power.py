# Copyright 2026 Leonid Purgalin
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# https://github.com/leonidpur/litnicelm

"""Host power as a checked benchmark condition, like the locked GPU clock."""

from ..measure.environment import host_power_state
from ..spec import EnvironmentSpec


class HostPowerCondition:
    """Require AC power and a fixed platform profile.

    preflight() refuses to start a run on a host that differs; row_flags()
    flags a measured process whose host state changed during the run.
    """

    def __init__(self, environment: EnvironmentSpec) -> None:
        self.require_ac_power = environment.require_ac_power
        self.platform_profile = environment.platform_profile

    def required_text(self) -> str:
        power = "AC" if self.require_ac_power else "any power source"
        return f"{power}, platform profile {self.platform_profile}"

    def mismatches(self, state: dict[str, object]) -> list[str]:
        """Human-readable differences between a host state and the spec."""
        found = []
        if self.require_ac_power and state.get("ac_online") is not True:
            found.append(f"AC online={state.get('ac_online')}")
        if state.get("platform_profile") != self.platform_profile:
            found.append(f"platform profile {state.get('platform_profile')}")
        return found

    def preflight(self) -> dict[str, object]:
        """Print the host power state; raise if it differs from the spec."""
        state = host_power_state()
        ac = {True: "AC on", False: "on battery"}.get(state["ac_online"], "AC unknown")
        print(
            f"Host power: {ac}, platform profile {state['platform_profile']}, "
            f"CPU governor {state['cpu_governor']}, "
            f"EPP {state['cpu_energy_performance_preference']} "
            f"(required: {self.required_text()})",
            flush=True,
        )
        problems = self.mismatches(state)
        if problems:
            raise RuntimeError(
                f"host power condition not met: {', '.join(problems)}; "
                f"required {self.required_text()}"
            )
        return state

    def row_flags(self, host_state: dict[str, object]) -> dict[str, bool]:
        """host_state_flagged for one measured process's host state."""
        return {"host_state_flagged": bool(self.mismatches(host_state))}
