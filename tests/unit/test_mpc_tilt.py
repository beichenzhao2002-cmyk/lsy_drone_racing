"""Regression checks for the MPC combined attitude command constraint."""

import numpy as np
import pytest
from casadi import Function
from crazyflow.dynamics import Dynamics, load_params

from lsy_drone_racing.control import attitude_mpc


def test_combined_tilt_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Combined roll and pitch must fit the physical tilt even inside the axis bounds."""
    monkeypatch.setattr(attitude_mpc, "AcadosOcpSolver", lambda *args, **kwargs: object())
    parameters = load_params(Dynamics.so_rpy, "cf21B_500")
    _, ocp = attitude_mpc.create_ocp_solver(
        0.75, 25, parameters, racing=True, tilt_limit=0.60, combined_tilt_limit=True
    )
    evaluate = Function("combined_tilt", [ocp.model.u], [ocp.model.con_h_expr])
    for roll, pitch, allowed in (
        (0.0, 0.0, True),
        (0.60, 0.0, True),
        (0.0, -0.60, True),
        (0.60, 0.60, False),
        (-0.60, 0.60, False),
        (0.30, 0.30, True),
    ):
        command = np.array([roll, pitch, 0.0, float(parameters["mass"]) * 9.81])
        value = float(evaluate(command))
        feasible = bool(ocp.constraints.lh[0] - 1e-12 <= value <= ocp.constraints.uh[0] + 1e-12)
        assert feasible is allowed
    np.testing.assert_allclose(ocp.constraints.ubu[:2], [0.60, 0.60])


def test_original_mpc_has_no_combined_tilt_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing example controllers keep their original solver constraints."""
    monkeypatch.setattr(attitude_mpc, "AcadosOcpSolver", lambda *args, **kwargs: object())
    parameters = load_params(Dynamics.so_rpy, "cf21B_500")
    _, ocp = attitude_mpc.create_ocp_solver(0.50, 25, parameters)
    assert ocp.model.con_h_expr == []
    np.testing.assert_allclose(ocp.constraints.ubu[:2], [0.50, 0.50])
