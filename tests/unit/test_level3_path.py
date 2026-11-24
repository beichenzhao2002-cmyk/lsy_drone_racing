import numpy as np
import pytest

from lsy_drone_racing.control.level3_mpc import _can_retain_path, _gate_exit_window
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator


@pytest.mark.parametrize(
    ("first_offset", "returns_to_gate", "direction", "retain"),
    [(0.76, True, 1, True), (0.76, True, -1, True),
     (0.76, False, 1, False), (0.12, True, 1, False)],
)
def test_path_gate_plane_intersections(first_offset, returns_to_gate, direction, retain):
    # Cross the plane, return on the outside of the frame, then enter the aperture.
    points = np.array([
        [-0.80, first_offset, 0.80], [0.80, first_offset, 0.80],
        [0.80, 0.80, 0.80], [-0.65, 0.80, 0.80],
        [-0.65, 0.00, 0.80], [0.68, 0.00, 0.80],
    ])
    if not returns_to_gate:
        points = points[:2]
    points[:, 0] *= direction
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    path = CubicHermiteSpline(np.r_[0.0, distances.cumsum()], points, np.zeros_like(points))
    observation = {
        "gates_pos": np.array([[0.0, 0.0, 0.80]]),
        "gates_quat": np.array([[0.0, 0.0, 0.0, 1.0]]),
        "obstacles_pos": np.empty((0, 3)),
        "n_gates_passed": 0,
        "gate_sequence": [0],
        "gate_sequence_direction": [direction],
    }
    assert _can_retain_path(path, 0.0, observation, 0.075) == retain

@pytest.mark.parametrize(("offset", "has_window"), [(0.0, True), (0.12, False)])
def test_gate_exit_window(offset, has_window):
    # The spline has a knot exactly on a rotated gate plane.
    center = np.array([1.37, -0.89, 0.8])
    axis = np.array([0.6, 0.8, 0.0])
    sideways = np.array([-0.8, 0.6, 0.0])
    arc = np.array([0.0, 0.4, 0.8])
    positions = center + (arc - 0.4)[:, None] * axis + offset * sideways
    path = CubicHermiteSpline(arc, positions, np.broadcast_to(axis, positions.shape))
    clock = PchipInterpolator(arc, arc)
    window = _gate_exit_window(path, clock, 0.8, center, axis)
    if has_window:
        crossing, exit_time = window
        assert crossing == pytest.approx(0.4)
        assert 0.49 < exit_time < 0.62
    else:
        assert window is None
