"""Opt-in simulation MPC with drag, offset estimation and individual rotor headroom.

The Handout supplies rigid-body relationships; the course model supplies nominal
actuator parameters. Command, body-tilt and rotor reserves are engineering
margins, not measured hardware limits. PPO weights are unchanged.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.utils.rotation import cs_rpy2matrix
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.attitude_mpc import create_acados_model
from lsy_drone_racing.control.level3_clearance import (
    ClearancePlanes,
    build_clearance_planes,
    set_clearance_planes,
)
from lsy_drone_racing.control.level3_curvature_controller import Level3CurvatureController

if TYPE_CHECKING:
    from acados_template import AcadosModel, AcadosOcp
    from ml_collections import ConfigDict

_PHYSICAL_SETTINGS = {
    "facets": 16,
    "aperture": 0.08,
    "body_limit": 0.55,
    "force_mapping": True,
    "rotor": True,
    "rti": True,
    "drag": True,
    "all_stages": True,
}


def reference_angles(acceleration: np.ndarray, parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a desired world-frame force into the existing attitude convention."""
    force = parameters["mass"] * (acceleration - np.asarray(parameters["gravity_vec"]))
    thrust = np.maximum(np.linalg.norm(force, axis=1), 1e-9)
    angles = np.zeros_like(force)
    angles[:, 0] = -np.arcsin(np.clip(force[:, 1] / thrust, -1.0, 1.0))
    angles[:, 1] = np.arctan2(force[:, 0], force[:, 2])
    return angles, thrust


class Level3PhysicalController(Level3CurvatureController):
    """Use the existing joint route with a physically coupled real-time MPC."""

    _clearance_plane_count = 6

    def __init__(self, obs: dict, info: dict, config: ConfigDict) -> None:
        """Build the opt-in physical model from nominal course actuator parameters."""
        self._physical_params = load_params(Dynamics.first_principles, config.sim.drone)
        physics = self._physical_params
        self._inertia = ca.DM(np.asarray(physics["J"]))
        lift = np.asarray(physics["rpm2thrust"])
        drag = np.asarray(physics["rpm2torque"])
        nominal = float(physics["mass"]) * 9.81 / 4.0
        roots = np.roots([lift[2], lift[1], lift[0] - nominal])
        rpm = float(max(x.real for x in roots if abs(x.imag) < 1e-8 and x.real > 0))
        ratio = float(np.polynomial.polynomial.polyval(rpm, drag)) / nominal
        mixing = np.vstack(
            (
                np.ones(4),
                np.diag([float(physics["L"]), float(physics["L"]), ratio])
                @ np.asarray(physics["mixing_matrix"]),
            )
        )
        self._mix_inverse = ca.DM(np.linalg.inv(mixing))
        self._motor_low = float(physics["thrust_min"])
        self._motor_high = float(physics["thrust_max"]) * 0.95
        super().__init__(obs, info, config)
        if not np.isclose(self._ocp.constraints.ubu[0], 0.60):
            raise ValueError("This candidate requires a 0.60 rad command envelope")
        if not np.isclose(self._ocp.constraints.ubu[3], 0.64):
            raise ValueError("This candidate requires a 0.64 N collective envelope")
        x, u, p = self._ocp.model.x, self._ocp.model.u, self._ocp.model.p
        self._motor_forces = ca.Function("motor_demand", [x, u, p], [self._motor_expression])
        self._motor_matrix = ca.Function(
            "motor_jacobian", [x, u, p], [ca.jacobian(self._motor_expression, u)]
        )
        self._motor_project = ca.conic(
            "project_motor",
            "osqp",
            {"h": ca.Sparsity.dense(4, 4), "a": ca.Sparsity.dense(20, 4)},
            {
                "error_on_fail": False,
                "osqp": {
                    "verbose": False,
                    "polish": True,
                    "eps_abs": 1e-7,
                    "eps_rel": 1e-7,
                    "max_iter": 2000,
                },
            },
        )

    def model_parameters(self) -> np.ndarray:
        """Share the estimated vertical offset with MPC and delay propagation."""
        return self._offset

    def _prepare_tracking_reference(
        self, reference: np.ndarray, acceleration: np.ndarray | None
    ) -> np.ndarray:
        """Map desired collective force through the identified thrust model."""
        if acceleration is not None:
            reference[:, 15] = np.clip(
                (reference[:, 15] - self.drone_params["acc_coef"])
                / self.drone_params["cmd_f_coef"],
                self._ocp.constraints.lbu[3],
                self._ocp.constraints.ubu[3],
            )
        return reference

    def center_on_gate(
        self, obs: dict, progress: int, action: np.ndarray, predictive: bool = False
    ) -> None:
        """Aperture correction lives in MPC instead of a post-solve action change."""

    def _prediction_model_factory(self, parameters: dict) -> AcadosModel:
        """Include nominal aerodynamic drag and the estimated vertical offset."""
        result = create_acados_model(parameters)
        disturbance = ca.MX.sym("tracking_offset", 3)
        result.p = disturbance
        f = result.f_expl_expr
        extra = disturbance
        if _PHYSICAL_SETTINGS.get("drag"):
            rotation = cs_rpy2matrix(result.x[3:6])
            extra = (
                extra
                + rotation
                @ ca.DM(np.asarray(self._physical_params["drag_matrix"]))
                @ rotation.T
                @ result.x[6:9]
                / parameters["mass"]
            )
        result.f_expl_expr = ca.vertcat(f[:6], f[6:9] + extra, f[9:])
        return result

    def _configure_prediction_ocp(self, ocp: AcadosOcp) -> None:
        """Couple command, future body tilt and rotor force constraints."""
        facets = 16
        directions = np.column_stack(
            (
                np.cos(np.arange(facets) * 2 * np.pi / facets),
                np.sin(np.arange(facets) * 2 * np.pi / facets),
            )
        )
        expression = ca.vertcat(
            *[float(row[0]) * ocp.model.u[0] + float(row[1]) * ocp.model.u[1] for row in directions]
        )
        ocp.model.con_h_expr_0 = expression
        ocp.constraints.lh_0 = np.full(facets, -1000.0)
        ocp.constraints.uh_0 = np.full(facets, 0.60 * np.cos(np.pi / facets))
        ocp.model.con_h_expr = expression
        ocp.constraints.lh = np.full(facets, -1000.0)
        ocp.constraints.uh = np.full(facets, 0.60 * np.cos(np.pi / facets))
        index = ocp.model.con_h_expr.rows()
        x, u = ocp.model.x, ocp.model.u
        ocp.model.con_h_expr = ca.vertcat(ocp.model.con_h_expr, ca.cos(x[3]) * ca.cos(x[4]))
        ocp.constraints.lh = np.r_[ocp.constraints.lh, np.cos(0.55)]
        ocp.constraints.uh = np.r_[ocp.constraints.uh, 1000.0]
        ocp.constraints.idxsh = np.r_[ocp.constraints.idxsh, index].astype(int)
        for name, coefficient in [("zl", 200.0), ("zu", 200.0), ("Zl", 4000.0), ("Zu", 4000.0)]:
            setattr(ocp.cost, name, np.r_[getattr(ocp.cost, name), coefficient])
        phi, theta = x[3], x[4]
        matrix = ca.vertcat(
            ca.horzcat(1, 0, -ca.sin(theta)),
            ca.horzcat(0, ca.cos(phi), ca.sin(phi) * ca.cos(theta)),
            ca.horzcat(0, -ca.sin(phi), ca.cos(phi) * ca.cos(theta)),
        )
        omega = matrix @ x[9:12]
        omega_dot = ca.jacobian(omega, x) @ ocp.model.f_expl_expr
        torque = self._inertia @ omega_dot + ca.cross(omega, self._inertia @ omega)
        forces = self._mix_inverse @ ca.vertcat(u[3], torque)
        self._motor_expression = forces
        for suffix in ("", "_0"):
            key = "con_h_expr" + suffix
            setattr(ocp.model, key, ca.vertcat(getattr(ocp.model, key), forces))
            for side, bound in (("lh", self._motor_low), ("uh", self._motor_high)):
                key = side + suffix
                setattr(
                    ocp.constraints, key, np.r_[getattr(ocp.constraints, key), np.full(4, bound)]
                )
        ocp.parameter_values = np.zeros(3)
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver_iter_max = 40

    def begin_race(self) -> None:
        """Clear race-local estimator state and disable post-solve altitude feedback."""
        super().begin_race()
        self._settings["altitude_feedback"] = False
        self._offset = np.zeros(3)
        self._offset_previous = None
        self._physical_counts = Counter()
        self._intrinsic_ticks = 0

    def track_reference(
        self,
        obs: dict,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray | None = None,
        yaw: float = 0.0,
    ) -> np.ndarray:
        """Compensate nominal drag and thrust bias before delay-aligned tracking."""
        previous = self.drone_params
        self.drone_params = dict(
            previous, gravity_vec=np.asarray(previous["gravity_vec"]) + self._offset
        )
        if _PHYSICAL_SETTINGS.get("drag") and acceleration is not None:
            angles, _ = reference_angles(acceleration, self.drone_params)
            angles[:, 2] = yaw
            rotation = Rotation.from_euler("xyz", angles).as_matrix()
            world_drag = np.einsum(
                "nij,jk,nlk->nil",
                rotation,
                np.asarray(self._physical_params["drag_matrix"]),
                rotation,
            )
            drag_acceleration = np.einsum("nij,nj->ni", world_drag, velocity) / previous["mass"]
            acceleration = acceleration - drag_acceleration
        try:
            return super().track_reference(obs, position, velocity, acceleration, yaw=yaw)
        finally:
            self.drone_params = previous

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Update the thrust estimator, shift the RTI warm start and solve."""
        up = float(Rotation.from_quat(obs["quat"]).as_matrix()[2, 2])
        previous = self._offset_previous
        force = float(self._last_command[3])
        p = self.drone_params
        if previous is not None:
            velocity, old_up, old_force, tick = previous
            interval = max(self._dt, (self._tick - tick) * self._dt)
            observed = (float(obs["vel"][2]) - velocity) / interval
            nominal = (p["acc_coef"] + p["cmd_f_coef"] * 0.5 * (force + old_force)) / p[
                "mass"
            ] * 0.5 * (up + old_up) + p["gravity_vec"][2]
            if _PHYSICAL_SETTINGS.get("drag"):
                rotation = Rotation.from_quat(obs["quat"]).as_matrix()
                drag_acceleration = (
                    rotation
                    @ np.asarray(self._physical_params["drag_matrix"])
                    @ rotation.T
                    @ obs["vel"]
                    / p["mass"]
                )
                nominal += drag_acceleration[2]
            residual = float(observed - nominal)
            if (
                obs["pos"][2] > 0.08
                and np.linalg.norm(obs["ang_vel"]) < 3.0
                and (abs(residual) < 6.0)
                and (abs(force - old_force) < 0.1)
            ):
                alpha = _PHYSICAL_SETTINGS.get("observer_gain", 0.1)
                self._offset[2] = np.clip(
                    (1 - alpha) * self._offset[2] + alpha * residual, -3.0, 3.0
                )
                self._physical_counts["observer_updates"] += 1
        self._offset_previous = (float(obs["vel"][2]), up, force, self._tick)
        for stage in range(self._N + 1):
            self._acados_ocp_solver.set(stage, "p", self._offset)
        if (
            _PHYSICAL_SETTINGS.get("rti")
            and self._tick
            and (getattr(self, "solver_status", None) in (0, 2))
        ):
            solver = self._acados_ocp_solver
            states = solver.get_flat("x").reshape(self._N + 1, self._nx)
            controls = solver.get_flat("u").reshape(self._N, self._nu)
            fraction = min(1.0, self._dt / self._prediction_dt)
            states[:-1] += fraction * np.diff(states, axis=0)
            controls[:-1] += fraction * np.diff(controls, axis=0)
            for stage in range(self._N + 1):
                solver.set(stage, "x", states[stage])
            for stage in range(self._N):
                solver.set(stage, "u", controls[stage])
        action = super().compute_control(obs, info)
        if _PHYSICAL_SETTINGS.get("rotor") and int(obs["n_gates_passed"]) < len(
            obs["gate_sequence"]
        ):
            state = self._acados_ocp_solver.get(0, "x")
            demands = np.asarray(self._motor_forces(state, action, self._offset)).ravel()
            if (
                not np.isfinite(demands).all()
                or np.min(demands) < self._motor_low - 2e-05
                or np.max(demands) > self._motor_high + 2e-05
            ):
                self._physical_counts["projection"] += 1
                constant = np.asarray(self._motor_forces(state, np.zeros(4), self._offset)).ravel()
                matrix = np.asarray(self._motor_matrix(state, np.zeros(4), self._offset))
                directions = np.column_stack(
                    (np.cos(np.arange(16) * np.pi / 8), np.sin(np.arange(16) * np.pi / 8))
                )
                poly = np.column_stack((directions, np.zeros((16, 2))))
                constraints = np.vstack((matrix, poly))
                lower = np.r_[np.full(4, self._motor_low) - constant, np.full(16, -1000.0)]
                upper = np.r_[
                    np.full(4, self._motor_high) - constant, np.full(16, 0.6 * np.cos(np.pi / 16))
                ]
                h = np.diag([20.0, 20.0, 1.0, 10.0])
                result = self._motor_project(
                    h=h,
                    g=-h @ action,
                    a=constraints,
                    lba=lower,
                    uba=upper,
                    lbx=self._ocp.constraints.lbu,
                    ubx=self._ocp.constraints.ubu,
                )
                projected = np.asarray(result["x"]).ravel()
                if (
                    np.isfinite(projected).all()
                    and np.max(constraints @ projected - upper) < 2e-05
                    and (np.max(lower - constraints @ projected) < 2e-05)
                ):
                    action[:] = projected
                else:
                    self._physical_counts["infeasible_projection"] += 1
        self._last_command = action
        return action

    def _set_obstacle_clearance(self, obs: dict, position: np.ndarray) -> None:
        """Use observed obstacles and gate aperture planes in the same optimization."""
        solver = self._acados_ocp_solver
        gates = obs["gates_visited"]
        poles = obs["obstacles_visited"]
        rotations = (
            Rotation.from_quat(obs["gates_quat"][gates]).as_matrix()
            if gates.any()
            else np.empty((0, 3, 3))
        )
        forecast = None
        if self._tick and getattr(self, "solver_status", None) in (0, 2):
            states = solver.get_flat("x").reshape(self._N + 1, self._nx)[:, :3]
            fraction = min(1.0, self._dt / self._prediction_dt)
            aligned = states[:-1] + fraction * np.diff(states, axis=0)
            if np.linalg.norm(aligned[0] - obs["pos"]) < 0.19:
                forecast = aligned[1:]
        original = build_clearance_planes(
            position,
            obs["gates_pos"][gates],
            rotations,
            obs["obstacles_pos"][poles],
            forecast=forecast,
        )
        normal = np.zeros((len(position), 6, 3))
        lower = np.zeros((len(position), 6))
        active = np.zeros((len(position), 6), bool)
        normal[:, :2] = original.normal
        lower[:, :2] = original.lower
        active[:, :2] = original.active
        progress = int(obs["n_gates_passed"])
        if (
            _PHYSICAL_SETTINGS.get("intrinsic", True)
            and (not self._retreating)
            and (progress >= _PHYSICAL_SETTINGS.get("start_after", 0))
        ):
            gate = obs["gate_sequence"][progress]
            rotation = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()
            forward, side, up = rotation.T
            forward *= obs["gate_sequence_direction"][progress]
            center = obs["gates_pos"][gate]
            offset = obs["pos"] - center
            along = offset @ forward
            forward_speed = obs["vel"] @ forward
            if obs["gates_visited"][gate] and -0.65 < along < -0.025 and (forward_speed > 0.3):
                lead = -along / forward_speed
                if 0.045 < lead < self._T_HORIZON - 0.06:
                    stage = int(
                        np.clip(round(lead / self._prediction_dt) - 1, 0, len(position) - 1)
                    )
                    if _PHYSICAL_SETTINGS.get("predicted_time") and forecast is not None:
                        signed = (aligned - center) @ forward
                        crossings = np.flatnonzero((signed[:-1] <= 0.0) & (signed[1:] > 0.0))
                        if len(crossings):
                            stage = int(np.clip(crossings[0], 0, len(position) - 1))
                    radius = _PHYSICAL_SETTINGS.get("window", 1)
                    for j in range(max(0, stage - radius), min(len(position), stage + radius + 1)):
                        for k, bearing in enumerate([side, -side, up, -up], 2):
                            normal[j, k] = bearing
                            lower[j, k] = bearing @ center - _PHYSICAL_SETTINGS.get(
                                "aperture", 0.08
                            )
                            active[j, k] = True
                    self._intrinsic_ticks += 1
        combined = ClearancePlanes(normal, lower, active)
        self._obstacle_planes_active = set_clearance_planes(
            solver, combined, self._nx, self._obstacle_planes_active
        )
