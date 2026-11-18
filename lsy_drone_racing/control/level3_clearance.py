"""Build supporting planes from the observed Level 3 gate and pole geometry."""

from dataclasses import dataclass

import numpy as np


_GATE_BAR_CENTERS = np.array([[0., 0., .28], [0., 0., -.28],
                              [0., .28, 0.], [0., -.28, 0.], [0., 0., -.86]])
_GATE_BAR_SIZES = np.array([[.01, .36, .08], [.01, .36, .08],
                                 [.01, .08, .36], [.01, .08, .36], [.05, .05, .5]])


@dataclass(frozen=True)
class ClearancePlanes:
    normal: np.ndarray
    lower: np.ndarray
    active: np.ndarray


def _box_planes(reference, centers, rotations):
    local = np.einsum('sgi,gij->sgj', reference[:, None] - centers, rotations)
    offset = local[:, :, None, :] - _GATE_BAR_CENTERS
    closest = np.clip(offset, -_GATE_BAR_SIZES, _GATE_BAR_SIZES)
    residual = offset - closest
    distance = np.linalg.norm(residual, axis=3)
    direction = residual / np.maximum(distance[..., None], 1e-8)
    # Give interior points a signed distance and a normal to the nearest exit face.
    stage, gate, bar = np.where(distance < 1e-8)
    if len(stage):
        remaining = _GATE_BAR_SIZES[bar] - np.abs(offset[stage, gate, bar])
        axis = np.argmin(remaining, axis=1)
        sign = np.where(offset[stage, gate, bar, axis] >= 0., 1., -1.)
        direction[stage, gate, bar] = 0.
        direction[stage, gate, bar, axis] = sign
        closest[stage, gate, bar, axis] = sign * _GATE_BAR_SIZES[bar, axis]
        distance[stage, gate, bar] = -remaining[np.arange(len(stage)), axis]
    normal = np.einsum('sgbi,gji->sgbj', direction, rotations)
    surface = centers[None, :, None] + np.einsum(
        'sgbi,gji->sgbj', closest + _GATE_BAR_CENTERS, rotations,
    )
    lower = np.sum(normal * surface, axis=-1)
    return normal.reshape(len(reference), -1, 3), lower.reshape(len(reference), -1), distance.reshape(len(reference), -1)


def _pole_planes(reference, poles, radius):
    offset = reference[:, None, :2] - poles[:, :2]
    distance = np.linalg.norm(offset, axis=2)
    direction = offset / np.maximum(distance[..., None], 1e-8)
    direction[distance < 1e-8] = [1., 0.]
    lower = np.sum(direction * poles[:, :2], axis=-1) + radius
    normal = np.concatenate((direction, np.zeros((*distance.shape, 1))), axis=2)
    return normal, lower, distance - radius


def build_clearance_planes(
    reference, gate_centers, gate_rotations, pole_centers, *, forecast=None,
    buffer=.095, activation=.19, max_planes=2, pole_radius=.015,
):
    """Select the closest supported obstacles at each point of the MPC horizon.

    The gate and pole inputs contain observed objects. Every active constraint
    has the form normal @ position >= lower. Use slack for the MPC constraints.
    The forecast uses the same future times as the reference.
    """
    reference = np.asarray(reference, dtype=float)
    gate_centers = np.asarray(gate_centers, dtype=float)
    gate_rotations = np.asarray(gate_rotations, dtype=float)
    pole_centers = np.asarray(pole_centers, dtype=float)
    if (reference.ndim != 2 or reference.shape[1] != 3
            or gate_centers.ndim != 2 or gate_centers.shape[1] != 3
            or gate_rotations.shape != (len(gate_centers), 3, 3)
            or pole_centers.ndim != 2 or pole_centers.shape[1] != 3
            or not all(np.isfinite(array).all() for array in (reference, gate_centers, gate_rotations, pole_centers))
            or not 0. <= buffer <= activation or max_planes < 1 or pole_radius <= 0.):
        raise ValueError("invalid observed obstacle geometry or clearance parameters")
    count = len(reference)
    output = ClearancePlanes(
        np.zeros((count, max_planes, 3)), np.zeros((count, max_planes)), np.zeros((count, max_planes), dtype=bool),
    )
    if not count or not (len(gate_centers) or len(pole_centers)):
        return output
    blocks = []
    if len(gate_centers):
        blocks.append(_box_planes(reference, gate_centers, gate_rotations))
    if len(pole_centers):
        blocks.append(_pole_planes(reference, pole_centers, pole_radius))
    normal, lower, signed_distance = (np.concatenate(group, axis=1) for group in zip(*blocks))
    urgency = signed_distance.copy()
    if forecast is not None:
        forecast = np.asarray(forecast, dtype=float)
        if forecast.shape != reference.shape or not np.isfinite(forecast).all():
            raise ValueError("forecast must match reference positions")
        residual = forecast - reference
        credible = np.linalg.norm(residual, axis=1) < .24
        forecast_distance = signed_distance + np.einsum('spi,si->sp', normal, residual)
        reachable = credible[:, None] & (signed_distance < activation + .13)
        urgency = np.where(reachable, np.minimum(urgency, forecast_distance), urgency)
    order = np.argsort(urgency, axis=1, kind='stable')[:, :max_planes]
    stage = np.arange(count)[:, None]
    active = urgency[stage, order] < activation
    width = order.shape[1]
    output.normal[:, :width, :] = np.where(active[..., None], normal[stage, order], 0.)
    output.lower[:, :width] = np.where(active, lower[stage, order] + buffer, 0.)
    output.active[:, :width] = active
    return output


_INACTIVE_BOUND = 1000.0


def add_soft_clearance(ocp, state_count, action_count, plane_count=2):
    """Allocate one side-specific slack per clearance row.

    The racing attitude MPC already has soft tilt bounds on intermediate nodes.
    Node zero pins the measured state and only allocates the general-row slacks.
    """
    ocp.constraints.C = np.zeros((plane_count, state_count))
    ocp.constraints.D = np.zeros((plane_count, action_count))
    ocp.constraints.lg = np.full(plane_count, -_INACTIVE_BOUND)
    ocp.constraints.ug = np.full(plane_count, _INACTIVE_BOUND)
    ocp.constraints.idxsg = np.arange(plane_count)
    for name, coefficient in (('zl', 260.), ('zu', 260.), ('Zl', 1400.), ('Zu', 1400.)):
        weight = np.full(plane_count, coefficient)
        setattr(ocp.cost, name, np.r_[getattr(ocp.cost, name), weight])
        setattr(ocp.cost, f'{name}_0', weight.copy())


def set_clearance_planes(solver, planes: ClearancePlanes, state_count, previously_active):
    """Replace active constraints and clear stale rows on intermediate nodes."""
    active_stages = planes.active.any(axis=1)
    if previously_active is None:
        previously_active = np.zeros_like(active_stages)
    plane_count = planes.active.shape[1]
    matrix = np.zeros((plane_count, state_count))
    for index in np.flatnonzero(active_stages | previously_active):
        matrix[:, :3] = planes.normal[index]
        lower = np.where(planes.active[index], planes.lower[index], -_INACTIVE_BOUND)
        stage = int(index) + 1
        solver.constraints_set(stage, 'C', matrix, api='new')
        solver.constraints_set(stage, 'lg', lower)
    return active_stages
