"""Level 3 controller for evaluating compensation of command transport delays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from casadi import Function
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates, rpy_rates2ang_vel
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.level3_racing_controller import Level3RacingController

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray


class Level3DelayController(Level3RacingController):
    """Evaluate control with one modeled period of command transport delay."""

    def __init__(
        self, obs: dict[str, NDArray[np.floating]], info: dict, config: ConfigDict
    ) -> None:
        """Prepare a rollout of the identified dynamics with the queued command."""
        super().__init__(obs, info, config)
        model = self._ocp.model
        if model.p.numel():
            dynamics = Function(
                "delay_model", [model.x, model.u, model.p], [model.f_expl_expr]
            )
            self._delay_dynamics = lambda state, command: dynamics(
                state, command, self.model_parameters()
            )
        else:
            self._delay_dynamics = Function(
                "delay_model", [model.x, model.u], [model.f_expl_expr]
            )

    def track_reference(
        self,
        obs: dict[str, NDArray[np.floating]],
        position: NDArray[np.floating],
        velocity: NDArray[np.floating],
        acceleration: NDArray[np.floating] | None = None,
        yaw: float = 0.0,
    ) -> NDArray[np.floating]:
        """Propagate the queued command and shift the MPC reference by the same time."""
        attitude = Rotation.from_quat(obs["quat"]).as_euler("xyz")
        rates = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        state = np.concatenate((obs["pos"], attitude, obs["vel"], rates))
        command = self._last_command
        delay = self._dt

        def derivative(value: NDArray[np.floating]) -> NDArray[np.floating]:
            return np.asarray(self._delay_dynamics(value, command)).ravel()

        k1 = derivative(state)
        k2 = derivative(state + 0.5 * delay * k1)
        k3 = derivative(state + 0.5 * delay * k2)
        k4 = derivative(state + delay * k3)
        predicted = state + delay / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        quat = Rotation.from_euler("xyz", predicted[3:6]).as_quat()
        forecast = dict(
            obs,
            pos=predicted[:3],
            quat=quat,
            vel=predicted[6:9],
            ang_vel=rpy_rates2ang_vel(quat, predicted[9:12]),
        )
        fraction = min(1.0, delay / self._prediction_dt)

        def shift(reference: NDArray[np.floating]) -> NDArray[np.floating]:
            shifted = reference.copy()
            shifted[:-1] += fraction * np.diff(reference, axis=0)
            return shifted

        position = shift(position)
        velocity = shift(velocity)
        acceleration = shift(acceleration) if acceleration is not None else None
        self._delay_reference = (forecast, position, velocity, acceleration)
        return super().track_reference(forecast, position, velocity, acceleration, yaw=yaw)

    def _feedback_reference(
        self, obs: dict, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Use the forecast for altitude and aperture feedback as well as MPC."""
        return self._delay_reference
