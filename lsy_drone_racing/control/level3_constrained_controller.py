"""Level 3 simulation controller with one-step delay and combined tilt constraints."""

from lsy_drone_racing.control.level3_delay_controller import Level3DelayController


class Level3ConstrainedController(Level3DelayController):
    """Constrain the MPC command to the same tilt cone as the actuator output."""

    combined_tilt_limit = True
