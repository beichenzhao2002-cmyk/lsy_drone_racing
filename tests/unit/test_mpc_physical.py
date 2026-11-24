"""Check offset compensation, motor headroom and recoverable attitude constraints."""

from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from lsy_drone_racing.control import attitude_mpc
from lsy_drone_racing.control.level3_physical_controller import Level3PhysicalController
from lsy_drone_racing.utils import load_config


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch) -> Level3PhysicalController:
    """Construct the real problem without compiling an acados shared library."""

    def backend(ocp: attitude_mpc.AcadosOcp, *_args: object, **_kwargs: object) -> SimpleNamespace:
        ocp.make_consistent()
        return SimpleNamespace(reset=lambda: None)

    monkeypatch.setattr(attitude_mpc, "AcadosOcpSolver", backend)
    config = load_config(Path("config/level3_mpc_curvature.toml"))
    return Level3PhysicalController({"pos": np.array([0.0, 0.0, 0.05])}, {}, config)


def test_bias_compensates_hover_in_both_predictors(controller: Level3PhysicalController) -> None:
    """MPC and queued-command propagation must see the same estimated disturbance."""
    parameters = controller.drone_params
    controller._offset[2] = 1.2
    force = parameters["mass"] * (9.81 - 1.2)
    command = np.array([0.0, 0.0, 0.0, force / parameters["cmd_f_coef"]])
    derivative = np.asarray(controller._delay_dynamics(np.zeros(12), command)).ravel()
    np.testing.assert_allclose(derivative[6:9], 0.0, atol=1e-12)
    reference = np.zeros((controller._N + 1, controller._ny))
    reference[:, 15] = force
    mapped = controller._prepare_tracking_reference(reference, np.zeros((controller._N + 1, 3)))
    np.testing.assert_allclose(mapped[:, 15], command[3], atol=1e-12)
    controller.begin_race()
    np.testing.assert_array_equal(controller.model_parameters(), np.zeros(3))


def test_rotor_limit_reserves_force_for_attitude_recovery(
    controller: Level3PhysicalController,
) -> None:
    """A feasible collective force can still saturate a motor during attitude braking."""
    state = np.zeros(12)
    state[3], state[9] = 0.5, 2.0
    command = np.array([-0.58, 0.0, 0.0, 0.64])
    forces = np.asarray(controller._motor_forces(state, command, np.zeros(3))).ravel()
    np.testing.assert_allclose(forces.sum(), command[3], atol=1e-12)
    assert forces.max() > controller._motor_high
    torque_only = forces - command[3] / 4
    center = (
        controller._motor_low + controller._motor_high - torque_only.max() - torque_only.min()
    ) / 2
    command[3] = 4 * center
    forces = np.asarray(controller._motor_forces(state, command, np.zeros(3))).ravel()
    assert forces.min() >= controller._motor_low
    assert forces.max() <= controller._motor_high


def test_first_command_and_future_body_tilt_have_separate_constraints(
    controller: Level3PhysicalController,
) -> None:
    """Bound the executed command without imposing a hard body-angle constraint on x0."""
    ocp = controller._ocp
    assert ocp.model.con_h_expr_0.numel() == 20
    assert ocp.model.con_h_expr.numel() == 21
    assert ocp.constraints.idxsh.tolist() == [16]
    evaluate = ca.Function(
        "physical_first", [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.con_h_expr_0[:16]]
    )
    command = np.array([0.6, 0.6, 0.0, 0.45])
    polygon = np.asarray(evaluate(np.zeros(12), command, np.zeros(3))).ravel()
    assert np.any(polygon > ocp.constraints.uh_0[:16])
    controller._offset[:] = 0.0
    assert ocp.solver_options.nlp_solver_type == "SQP_RTI"
    assert controller._settings["altitude_feedback"] is False
