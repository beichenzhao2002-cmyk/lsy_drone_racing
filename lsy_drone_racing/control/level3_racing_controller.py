"""Level 3 MPC entry point for the original course configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.level3_mpc import Level3MPC

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray


_RACING_SETTINGS = {
    "soft_obstacle_clearance": True,
    "continuous_finish_retry": True,
    "spatial_detour_pruning": True,
    "hold_gate_exit": True,
    "cruise_speed": 4.9,
    "acceleration": 4.7,
    "lateral_acceleration": 3.4,
    "clearance": 0.39,
    "alignment": 0.4,
    "altitude_feedback": True,
    "aperture_offset_limit": 0.055,
    "short_alignment": 0.17,
    "aperture_feedback": True,
    "optimize_race_time": True,
    "compact_gate_search": True,
    "narrow_graph_search": True,
    "prefer_narrow_route": True,
    "prediction_seconds": 0.75,
    "acceleration_budget": 5.0,
    # Engineering margins for lab-oriented simulation. The Handout does not specify these limits.
    "mpc_tilt_limit": 0.60,
    "mpc_thrust_fraction": 0.80,
    "retain_clearance": 0.125,
    "vector_speed_profile": True,
    "recover_finish_gate": True,
    "straight_power": 0.8,
    "open_lateral_acceleration": 5.1,
    "predictive_race_tracking": True,
    "predictive_start_after": 1,
    "phase_correction": 0.32,
    "phase_lag_limit": 0.006,
}


class Level3RacingController(Level3MPC):
    """Track the course with a bounded Crazyflie thrust and tilt reference."""

    def __init__(
        self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict
    ) -> None:
        """Keep racing settings in the controller and leave the shared config intact."""
        controller_config = config.copy_and_resolve_references()
        controller_config.controller.update(_RACING_SETTINGS)
        super().__init__(obs, info, controller_config)

    def track_reference(
        self,
        obs: dict[str, NDArray[np.floating]],
        position: NDArray[np.floating],
        velocity: NDArray[np.floating],
        acceleration: NDArray[np.floating] | None = None,
        yaw: float = 0.0,
    ) -> NDArray[np.floating]:
        """Project the requested force into the available thrust and tilt envelope.

        Section 2.2 of the course Quadrotor Introduction Handout gives
        the force along the body z axis and the world-frame gravity term.
        Preserve the vertical component where possible and scale horizontal force.
        """
        if acceleration is not None:
            gravity = self.drone_params["gravity_vec"]
            mass = self.drone_params["mass"]
            min_thrust = 4 * self.drone_params["thrust_min"]
            max_thrust = self._ocp.constraints.ubu[3]
            force = mass * (acceleration - gravity)
            vertical = np.clip(force[:, 2], min_thrust, max_thrust)
            horizontal_limit = np.minimum(
                vertical * np.tan(self._ocp.constraints.ubu[0]),
                np.sqrt(np.maximum(max_thrust**2 - vertical**2, 0.0)),
            )
            horizontal = force[:, :2]
            magnitude = np.linalg.norm(horizontal, axis=1)
            horizontal *= np.minimum(1.0, horizontal_limit / np.maximum(magnitude, 1e-12))[:, None]
            force[:, 2] = vertical
            acceleration = force / mass + gravity
        return super().track_reference(obs, position, velocity, acceleration, yaw=yaw)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Enforce the command envelope after the gate and altitude corrections."""
        action = super().compute_control(obs, info)
        lower, upper = self._ocp.constraints.lbu, self._ocp.constraints.ubu
        np.clip(action, lower, upper, out=action)
        roll, pitch = action[:2]
        body_z_horizontal = np.array([np.cos(roll) * np.sin(pitch), -np.sin(roll)])
        tilt_sine = np.linalg.norm(body_z_horizontal)
        max_tilt_sine = np.sin(upper[0])
        if tilt_sine > max_tilt_sine:
            x, y = body_z_horizontal * (max_tilt_sine / tilt_sine)
            action[0] = -np.arcsin(y)
            action[1] = np.arcsin(x / np.cos(action[0]))
        return action
