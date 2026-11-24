"""Level 3 simulation controller with curvature-aware MPC constraint handling."""

from lsy_drone_racing.control.level3_constrained_controller import Level3ConstrainedController


class Level3CurvatureController(Level3ConstrainedController):
    """Use constraint Hessians while retaining the existing trajectory settings."""

    constraint_curvature = True
