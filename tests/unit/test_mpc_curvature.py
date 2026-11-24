"""Check curvature-aware MPC constraints and compatibility with the existing solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from casadi import Function, jacobian
from crazyflow.dynamics import Dynamics, load_params

from lsy_drone_racing.control import attitude_mpc

if TYPE_CHECKING:
    from acados_template import AcadosOcp


def _problem(monkeypatch: pytest.MonkeyPatch, **options: bool) -> AcadosOcp:
    monkeypatch.setattr(attitude_mpc, "AcadosOcpSolver", lambda *args, **kwargs: object())
    _, ocp = attitude_mpc.create_ocp_solver(
        0.75, 25, load_params(Dynamics.so_rpy, "cf21B_500"),
        racing=True, soft_obstacle_clearance=True, combined_tilt_limit=True,
        tilt_limit=0.60, **options,
    )
    return ocp


def test_curvature_preserves_command_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing the redundant upper side must retain the same physical command cone."""
    ocp = _problem(monkeypatch, constraint_curvature=True)
    assert ocp.solver_options.hessian_approx == "EXACT"
    assert ocp.solver_options.exact_hess_dyn == 0
    assert ocp.solver_options.exact_hess_constr == 1
    assert ocp.constraints.uh[0] > 1.0
    np.testing.assert_array_equal(ocp.constraints.ubu[:2], [0.60, 0.60])
    assert ocp.model.con_h_expr_0 == []


def test_initial_command_matches_the_future_cone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The action being executed must obey the same cone as later predicted commands."""
    ocp = _problem(monkeypatch, constraint_curvature=True, initial_command_constraint=True)
    ocp.make_consistent()
    first = Function("first_cone", [ocp.model.u], [ocp.model.con_h_expr_0])
    future = Function("future_cone", [ocp.model.u], [ocp.model.con_h_expr])
    for command, allowed in (
        ([0.60, 0.0, 0.0, 0.50], True),
        ([0.60, 0.60, 0.0, 0.50], False),
        ([0.30, 0.30, 0.0, 0.50], True),
    ):
        value = float(first(command))
        assert bool(value >= ocp.constraints.lh_0[0] - 1e-12) is allowed
        assert value == float(future(command))
    np.testing.assert_array_equal(ocp.constraints.lh_0, ocp.constraints.lh)


def test_initial_cone_allows_measured_attitude_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constraining the command must not make an observed tilt excursion infeasible."""
    ocp = _problem(monkeypatch, constraint_curvature=True, initial_command_constraint=True)
    derivative = Function(
        "state_derivative", [ocp.model.x, ocp.model.u],
        [jacobian(ocp.model.con_h_expr_0, ocp.model.x)],
    )
    state = np.zeros(12)
    state[3:5] = [0.9, 0.2]
    np.testing.assert_array_equal(
        np.asarray(derivative(state, [0.0, 0.0, 0.0, 0.5])), np.zeros((1, 12)),
    )
    ocp.make_consistent()
    assert ocp.cost.zl_0.shape == (2,)
    assert ocp.cost.zl.shape == (5,)


def test_legacy_combined_solver_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing controller classes retain their former numerical configuration."""
    ocp = _problem(monkeypatch)
    assert ocp.model.con_h_expr_0 == []
    assert ocp.solver_options.hessian_approx == "GAUSS_NEWTON"
    np.testing.assert_array_equal(ocp.constraints.uh, [1.0])


@pytest.mark.parametrize(
    "options",
    [
        {"constraint_curvature": True},
        {"initial_command_constraint": True},
    ],
)
def test_inconsistent_solver_modes_are_rejected(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, bool],
) -> None:
    """Fail before constructing a solver with incompatible nonlinear constraints."""
    monkeypatch.setattr(attitude_mpc, "AcadosOcpSolver", lambda *args, **kwargs: object())
    with pytest.raises(ValueError):
        attitude_mpc.create_ocp_solver(
            0.75, 25, load_params(Dynamics.so_rpy, "cf21B_500"), **options,
        )
