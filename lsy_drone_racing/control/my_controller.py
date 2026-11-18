"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class MyController(Controller):
    """State controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._last_gate_status = None

        # Same waypoints as in the attitude controller. Determined by trial and error.
        start_pos = obs["pos"]

        passed = int(obs["n_gates_passed"])
        gate_id = int(obs["gate_sequence"][passed])
        gate_pos = obs["gates_pos"][gate_id]
        gate_detected = bool(obs["gates_visited"][gate_id])

        print("Gates passed:", passed)
        print("Next gate ID:", gate_id)
        print("Next gate position:", gate_pos)
        print("Real position detected:", gate_detected)




        waypoints = [np.asarray(start_pos, dtype=float)]

        for i, gate_id in enumerate(obs["gate_sequence"]):
            gate_id = int(gate_id)
            gate_pos = np.asarray(obs["gates_pos"][gate_id])
            gate_quat = obs["gates_quat"][gate_id]
            direction = int(obs["gate_sequence_direction"][i])

            gate_axis = R.from_quat(gate_quat).apply([1.0, 0.0, 0.0])
            flight_direction = gate_axis * direction

            waypoints.append(gate_pos - 0.3 * flight_direction)
            waypoints.append(gate_pos)
            waypoints.append(gate_pos + 0.3 * flight_direction)

        waypoints = np.asarray(waypoints, dtype=float)
        print("Generated waypoints:", waypoints.shape)

        self._t_total = 22.5  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)

        self._tick = 0
        self._finished = False


    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone."""

        passed = int(obs["n_gates_passed"])
        gate_sequence = obs["gate_sequence"]

        if passed < len(gate_sequence):
            gate_id = int(gate_sequence[passed])
            detected = bool(obs["gates_visited"][gate_id])
            status = (passed, gate_id, detected)

            if status != self._last_gate_status:
                gate_pos = obs["gates_pos"][gate_id]
                gate_quat = obs["gates_quat"][gate_id]
                passing_direction = int(obs["gate_sequence_direction"][passed])

                print(f"Target gate: {gate_id}, detected: {detected}")
                print(f"Gate position: {gate_pos}")
                print(f"Gate orientation: {gate_quat}")
                print(f"Passing direction: {passing_direction}")

                gate_axis = R.from_quat(gate_quat).apply([1.0, 0.0, 0.0])
                flight_direction = gate_axis * passing_direction

                approach_pos = gate_pos - 0.3 * flight_direction
                exit_pos = gate_pos + 0.3 * flight_direction

                print(f"Approach point A: {approach_pos}")
                print(f"Gate center B: {gate_pos}")
                print(f"Exit point C: {exit_pos}")

                self._last_gate_status = status

        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True

        des_pos = self._des_pos_spline(t)
        des_yaw = 0.0
        des_quat = R.from_euler("z", des_yaw).as_quat()
        action = np.concatenate(
            (des_pos, np.zeros(6), des_quat, np.zeros(3)),
            dtype=np.float32
        )
        return action


    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
