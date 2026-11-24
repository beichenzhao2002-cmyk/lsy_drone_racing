"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from acados_template.builders import ocp_get_default_cmake_builder
from casadi import cos
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy import symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.level3_clearance import add_soft_clearance

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from the symbolic drone dynamics."""
    # For more info on the models, check out https://github.com/learnsyslab/crazyflow
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False, *, racing: bool = False,
    tilt_limit: float = 0.50, soft_obstacle_clearance: bool = False,
    combined_tilt_limit: bool = False, constraint_curvature: bool = False,
    initial_command_constraint: bool = False, clearance_plane_count: int = 2,
    model_factory: Callable[[dict], AcadosModel] | None = None,
    configure: Callable[[AcadosOcp], None] | None = None,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Create the solver with opt-in constraint curvature and initial command bounds.

    Legacy settings remain the default. The initial command option copies the
    combined command cone to stage zero without constraining the measured state.
    """
    if initial_command_constraint and not constraint_curvature:
        raise ValueError("Initial command constraint requires constraint curvature")
    if constraint_curvature and not combined_tilt_limit:
        raise ValueError("Constraint curvature requires the combined tilt model")
    ocp = AcadosOcp()

    # Set model
    ocp.model = (model_factory or create_acados_model)(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            50.0,  # pos
            50.0,  # pos
            400.0,  # pos
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # vel
            10.0,  # vel
            10.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            50.0,  # thrust
        ]
    )

    if racing:
        Q = np.diag([180., 180., 480., 2., 2., 2., 14., 14., 16., 2., 2., 2.])
        R = np.diag([1., 1., 1., 30.])
    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references. We will overwrite these later to track the trajectory
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-tilt_limit, -tilt_limit, -0.5])
    ocp.constraints.ubx = np.array([tilt_limit, tilt_limit, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    if racing:
        # Keep hard input limits, but permit recovering after an observed tilt excursion.
        ocp.constraints.idxsbx = np.arange(3)
        ocp.cost.zl = ocp.cost.zu = np.full(3, 40.)
        ocp.cost.Zl = ocp.cost.Zu = np.full(3, 200.)
        if soft_obstacle_clearance:
            if clearance_plane_count == 2:
                add_soft_clearance(ocp, nx, nu)
            else:
                add_soft_clearance(ocp, nx, nu, plane_count=clearance_plane_count)

    # Set Input Constraints (rpy < 30°)
    total_thrust_min = parameters["thrust_min"] * 4
    total_thrust_max = parameters["thrust_max"] * 4
    ocp.constraints.lbu = np.array([-tilt_limit, -tilt_limit, -0.5, total_thrust_min])
    ocp.constraints.ubu = np.array([tilt_limit, tilt_limit, 0.5, total_thrust_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    if combined_tilt_limit:
        # Bound combined commanded tilt; actual attitude can still overshoot.
        # Include simultaneous roll and pitch commands in this cone.
        roll, pitch = ocp.model.u[0], ocp.model.u[1]
        ocp.model.con_h_expr = cos(roll) * cos(pitch)
        ocp.constraints.lh = np.array([np.cos(tilt_limit)])
        # The cosine upper bound is redundant. Leaving it active at level
        # attitude makes the constraint Hessian numerically degenerate.
        ocp.constraints.uh = np.array([1000.0 if constraint_curvature else 1.0])
        if initial_command_constraint:
            # Also constrain the command that will actually be executed.
            # Acados does not copy path nonlinear constraints to stage zero.
            ocp.model.con_h_expr_0 = ocp.model.con_h_expr
            ocp.constraints.lh_0 = ocp.constraints.lh.copy()
            ocp.constraints.uh_0 = ocp.constraints.uh.copy()

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    if constraint_curvature:
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.exact_hess_dyn = 0
        ocp.solver_options.exact_hess_constr = 1
    else:
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    cmake_builder = None
    # On Windows, use CMake with MinGW to build the generated MPC solver DLL.
    if os.name == "nt":
        cmake_builder = ocp_get_default_cmake_builder()
        cmake_builder.generator = "MinGW Makefiles"
        cmake_builder.build_dir = "build-mingw"

    if configure is not None:
        configure(ocp)

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
        cmake_builder=cmake_builder,
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    racing_tracking = False
    combined_tilt_limit = False
    constraint_curvature = False
    initial_command_constraint = False
    _clearance_plane_count = 2
    _prediction_model_factory = None
    _configure_prediction_ocp = None

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt
        if self.racing_tracking:
            self._T_HORIZON = config.controller.get("prediction_seconds", self._T_HORIZON)
        self._prediction_dt = self._T_HORIZON / self._N

        # Same waypoints as in the state controller. Determined by trial and error.
        start_pos = obs["pos"]
        waypoints = np.array(
            [
                start_pos,
                [-1.0, 0.75, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [1.2, -0.15, 1.2],
                [1.05, 0.75, 1.2],
                [0.25, 1.25, 1.2],
            ]
        )
        self._t_total = 18.75  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = self._des_pos_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
        self._waypoints_vel = self._des_vel_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0

        self.drone_params = load_params(Dynamics.so_rpy, config.sim.drone)
        if self.racing_tracking and "mpc_thrust_fraction" in config.controller:
            fraction = config.controller.mpc_thrust_fraction
            thrust_max = self.drone_params["thrust_max"] * fraction
            hover_thrust = -self.drone_params["mass"] * self.drone_params["gravity_vec"][2]
            if not 0.0 < fraction <= 1.0 or 4 * thrust_max <= hover_thrust:
                raise ValueError("MPC collective thrust limit must exceed hover thrust")
            self.drone_params = dict(self.drone_params, thrust_max=thrust_max)
        self._soft_obstacle_clearance = self.racing_tracking and config.controller.get(
            "soft_obstacle_clearance", False,
        )
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params, racing=self.racing_tracking,
            tilt_limit=(
                config.controller.get("mpc_tilt_limit", 0.50) if self.racing_tracking else 0.50
            ),
            soft_obstacle_clearance=self._soft_obstacle_clearance,
            combined_tilt_limit=self.combined_tilt_limit,
            constraint_curvature=self.constraint_curvature,
            initial_command_constraint=self.initial_command_constraint,
            clearance_plane_count=self._clearance_plane_count,
            model_factory=self._prediction_model_factory,
            configure=self._configure_prediction_ocp,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._config = config
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        horizon = slice(i, i + self._N + 1)
        return self.track_reference(
            obs, self._waypoints_pos[horizon], self._waypoints_vel[horizon],
            yaw=self._waypoints_yaw[horizon],
        )

    def track_reference(
        self,
        obs: dict[str, NDArray[np.floating]],
        position: NDArray[np.floating],
        velocity: NDArray[np.floating],
        acceleration: NDArray[np.floating] | None = None,
        yaw: float = 0.0,
    ) -> NDArray[np.floating]:
        """Track an external horizon with optional acceleration feedforward."""
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        rates = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        state = np.concatenate((obs["pos"], rpy, obs["vel"], rates))
        solver = self._acados_ocp_solver
        solver.set(0, "lbx", state)
        solver.set(0, "ubx", state)

        reference = np.zeros((self._N + 1, self._ny))
        reference[:, :3] = position
        reference[:, 5] = yaw
        reference[:, 6:9] = velocity
        mass = self.drone_params["mass"]
        gravity = self.drone_params["gravity_vec"]
        reference[:, 15] = mass * -gravity[-1]
        if acceleration is not None:
            force = mass * (acceleration - gravity)
            thrust = np.maximum(np.linalg.norm(force, axis=1), 1e-6)
            reference[:, 3] = -np.arcsin(np.clip(force[:, 1] / thrust, -1.0, 1.0))
            reference[:, 4] = np.arctan2(force[:, 0], force[:, 2])
            reference[:, 12:15] = reference[:, 3:6]
            reference[:, 15] = thrust

        reference = self._prepare_tracking_reference(reference, acceleration)
        for stage in range(self._N):
            solver.set(stage, "yref", reference[stage])
        solver.set(self._N, "yref", reference[-1, :self._ny_e])
        self.solver_status = solver.solve()
        return solver.get(0, "u")

    def _prepare_tracking_reference(
        self, reference: np.ndarray, acceleration: np.ndarray | None
    ) -> np.ndarray:
        """Optional model-specific reference mapping; preserve the default reference."""
        return reference

    def model_parameters(self) -> np.ndarray:
        """Runtime model parameters, empty for the original identified model."""
        return np.zeros(int(self._ocp.model.p.numel()))

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
