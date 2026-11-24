"""Joint Level 3 gate planning and acados trajectory tracking."""

import numpy as np
from scipy.interpolate import BPoly, CubicHermiteSpline, PchipInterpolator
from scipy.optimize import brentq, minimize
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.attitude_mpc import AttitudeMPC
from lsy_drone_racing.control.level3_clearance import (
    _GATE_BAR_CENTERS,
    _GATE_BAR_SIZES,
    build_clearance_planes,
    set_clearance_planes,
)
from lsy_drone_racing.control.route_geometry import approach_route


def aperture_offsets(start, centers, sideways, limit):
    """Jointly shorten the gate-to-gate route within lateral aperture margins."""
    def objective(offsets):
        crossings = centers + sideways * offsets[:, None]
        segments = np.diff(np.vstack((start, crossings)), axis=0)
        lengths = np.maximum(np.linalg.norm(segments, axis=1), 1e-6)
        direction = segments / lengths[:, None]
        gradient = direction.copy()
        gradient[:-1] -= direction[1:]
        return (lengths.sum() + 4. * np.dot(offsets, offsets),
                np.sum(gradient * sideways, axis=1) + 8. * offsets)

    result = minimize(objective, np.zeros(len(centers)), jac=True,
                      bounds=[(-limit, limit)] * len(centers), method="L-BFGS-B",
                      options={"maxiter": 12})
    return centers + sideways * result.x[:, None] if result.success else centers


def race_waypoints(
    obs, clearance, alignment, aperture_margin=0.0, short_alignment=0.0, crossing_slope=0.0,
    local_endpoint_clearance=False, minimal_gate_knots=False,
):
    """Connect the remaining gates through aligned apertures and detours."""
    centers = obs["gates_pos"]
    rotations = Rotation.from_quat(obs["gates_quat"])
    axes = rotations.apply(np.broadcast_to([1., 0., 0.], centers.shape))
    sides = rotations.apply(np.broadcast_to([0., 0.32, 0.], centers.shape))
    obstacles = np.vstack((obs["obstacles_pos"], centers - sides, centers + sides))
    sequence = obs["gate_sequence"]
    progress = int(obs["n_gates_passed"])
    points = [obs["pos"]]
    headings = [None]
    if progress:
        previous = int(sequence[progress - 1])
        axis = axes[previous] * obs["gate_sequence_direction"][progress - 1]
        ahead = np.dot(obs["pos"] - centers[previous], axis)
        if 0 <= ahead < alignment - 0.05:
            points.append(centers[previous] + alignment * axis)
            headings.append(axis)

    remaining = sequence[progress:]
    crossings = centers[remaining]
    if aperture_margin > 0:
        crossings = aperture_offsets(points[-1], crossings, sides[remaining] / 0.32, aperture_margin)
    course = np.vstack((points[-1], crossings))
    segments = np.diff(course, axis=0)
    directions = segments / np.maximum(np.linalg.norm(segments, axis=1, keepdims=True), 1e-6)
    for index, (gate, direction) in enumerate(zip(remaining, obs["gate_sequence_direction"][progress:])):
        center = crossings[index]
        forward = axes[gate] * direction
        if crossing_slope > 0:
            # Preserve the crossing direction while allowing a bounded lateral component
            # toward the next leg. Keep height fixed through the aperture.
            lateral_axis = sides[gate] / 0.32
            approach = directions[index]
            departure = directions[index + 1] if index + 1 < len(remaining) else approach
            lateral = np.clip((approach + departure) @ lateral_axis / 2,
                              -crossing_slope, crossing_slope)
            forward = forward + lateral * lateral_axis
            forward /= np.linalg.norm(forward)
        entry_length = exit_length = alignment
        if short_alignment > 0:
            entry_blend = np.clip((directions[index] @ forward - 0.78) / 0.20, 0., 1.)
            entry_length -= (alignment - short_alignment) * entry_blend
            if index + 1 < len(remaining):
                exit_blend = np.clip((directions[index + 1] @ forward - 0.78) / 0.20, 0., 1.)
                exit_length -= (alignment - short_alignment) * exit_blend
        if minimal_gate_knots:
            entry_length = 0.
            exit_length = 0.25 if index + 1 == len(remaining) else 0.
        entry = center - entry_length * forward
        # Preserve the current pass when a measurement arrives inside the approach leg.
        relative = points[-1] - center
        along = np.dot(relative, forward)
        if not (-entry_length < along < 0 and np.linalg.norm(relative - along * forward) < 0.13):
            detour = approach_route(
                points[-1], entry, obstacles, clearance,
                local_endpoint_clearance=local_endpoint_clearance,
            )
            points.extend(detour)
            headings.extend([None] * len(detour))
            points.append(entry)
            headings.append(forward)
        points.extend((center, center + exit_length * forward))
        headings.extend((forward, forward))

    points = np.asarray(points)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.015]
    return points[keep], [heading for heading, retain in zip(headings, keep) if retain]


def _open_turn_weight(position, derivative, arc, obs):
    """Fade in extra cornering acceleration away from frames, poles and climbs."""
    gate_distances = np.linalg.norm(position[:, None] - obs["gates_pos"], axis=2)
    weight = np.clip((gate_distances.min(axis=1) - 0.50) / 0.28, 0., 1.)
    poles = obs["obstacles_pos"]
    if len(poles):
        pole_distances = np.linalg.norm(position[:, None, :2] - poles[:, :2], axis=2)
        weight *= np.clip((pole_distances.min(axis=1) - 0.31) / 0.18, 0., 1.)
    grade = np.abs(derivative[:, 2]) / np.maximum(np.linalg.norm(derivative, axis=1), 0.05)
    weight *= np.clip((0.45 - grade) / 0.30, 0., 1.)
    return weight * np.clip((arc - 0.20) / 0.25, 0., 1.)


def timed_trajectory(
    points, headings, velocity, speed, accel, lateral_accel, acceleration_budget=None,
    vector_speed_profile=False, straight_acceleration=0., open_turn=None,
):
    """Fit a smooth path and time it with curvature and acceleration limits."""
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    knots = np.r_[0., np.cumsum(lengths)]
    unit = delta / lengths[:, None]
    tangents = np.vstack((unit[0], (unit[:-1] + unit[1:]) / 2, unit[-1]))
    initial_speed = np.linalg.norm(velocity)
    if initial_speed > 0.2 and np.dot(unit[0], velocity) > 0.1:
        tangents[0] = (unit[0] + velocity / initial_speed) / 2
    for i, heading in enumerate(headings):
        if heading is not None:
            tangents[i] = heading
    path = CubicHermiteSpline(knots, points, tangents)
    return time_path(
        path, velocity, speed, accel, lateral_accel, acceleration_budget,
        vector_speed_profile, straight_acceleration, open_turn,
    )


def time_path(
    path, velocity, speed, accel, lateral_accel, acceleration_budget=None,
    vector_speed_profile=False, straight_acceleration=0., open_turn=None,
):
    """Time a position spline with curvature and acceleration limits."""
    knots = path.x
    samples = np.unique(np.r_[np.linspace(0., knots[-1], int(knots[-1] / 0.025) + 2), knots])
    position = path(samples)
    distances = np.linalg.norm(np.diff(position, axis=0), axis=1)
    derivative = path(samples, 1)
    curvature = np.linalg.norm(np.cross(derivative, path(samples, 2)), axis=1)
    curvature /= np.maximum(np.linalg.norm(derivative, axis=1), 0.05) ** 3
    open_weight = None
    if open_turn is not None:
        obs, open_limit = open_turn
        open_weight = _open_turn_weight(position, derivative, samples, obs)
        lateral_accel = lateral_accel + (open_limit - lateral_accel) * open_weight
    speeds = np.minimum(speed, np.sqrt(lateral_accel / np.maximum(curvature, 1e-5)))
    speeds[0] = np.clip(np.dot(velocity, derivative[0]), 0., speed)
    speeds[-1] = 0.
    if acceleration_budget is not None:
        # Centripetal acceleration consumes part of the thrust available to
        # accelerate or brake. Compute a conservative tangential bound per edge.
        turn_accel = curvature * speeds ** 2
        edge_turn = np.maximum(turn_accel[:-1], turn_accel[1:])
        if open_weight is not None:
            acceleration_budget = acceleration_budget + 2 * np.minimum(open_weight[:-1], open_weight[1:])
        tangential = np.minimum(accel, np.sqrt(np.maximum(acceleration_budget ** 2 - edge_turn ** 2, 0.15)))
    else:
        tangential = np.full(len(distances), accel)
    if straight_acceleration:
        # Apply additional tangential power where the path is already level
        # and straight. Leave braking and curve limits at their original values.
        unit_velocity = derivative / np.maximum(np.linalg.norm(derivative, axis=1, keepdims=True), 0.05)
        curve_factor = np.clip((0.8 - curvature) / 0.65, 0., 1.)
        grade_factor = np.clip((0.42 - np.abs(unit_velocity[:, 2])) / 0.30, 0., 1.)
        power = curve_factor * grade_factor
        drive = tangential + straight_acceleration * np.minimum(power[:-1], power[1:])
    else:
        drive = tangential
    if vector_speed_profile:
        # In squared speed, every acceleration bound is an additive distance budget.
        reach = np.r_[0., np.cumsum(2. * drive * distances)]
        forward = np.minimum.accumulate(speeds ** 2 - reach) + reach
        braking = np.r_[0., np.cumsum(2. * tangential * distances)] if straight_acceleration else reach
        backward = np.minimum.accumulate((forward + braking)[::-1])[::-1] - braking
        speeds = np.sqrt(np.maximum(backward, 0.))
    else:
        for i, ds in enumerate(distances):
            speeds[i + 1] = min(speeds[i + 1], np.sqrt(speeds[i] ** 2 + 2 * drive[i] * ds))
        for i in range(len(distances) - 1, 0, -1):
            reachable = np.sqrt(speeds[i + 1] ** 2 + 2 * tangential[i] * distances[i])
            speeds[i] = min(speeds[i], reachable)
    elapsed = np.r_[0., np.cumsum(2 * distances / np.maximum(speeds[:-1] + speeds[1:], 0.05))]
    return path, PchipInterpolator(elapsed, samples), elapsed[-1]


def obstacle_distances(samples, obs):
    """Compute clearance from the observed gate collision boxes and obstacle poles."""
    gate_rotation = Rotation.from_quat(obs["gates_quat"]).as_matrix()
    offset = samples[:, None] - obs["gates_pos"]
    local = np.einsum("sgi,gij->sgj", offset, gate_rotation)
    box_offset = np.abs(local[:, :, None] - _GATE_BAR_CENTERS) - _GATE_BAR_SIZES
    distances = np.linalg.norm(np.maximum(box_offset, 0.), axis=-1)
    minimum = distances.min(axis=(1, 2))
    poles = obs["obstacles_pos"]
    if len(poles):
        horizontal = np.linalg.norm(samples[:, None, :2] - poles[:, :2], axis=-1)
        minimum = np.minimum(minimum, (horizontal - 0.015).min(axis=1))
    return minimum


def clearance_deficit(path, obs, margin):
    """Sample spline geometry before accepting a faster route."""
    length = path.x[-1]
    samples = path(np.linspace(0., length, max(2, int(np.ceil(length / 0.035)) + 1)))
    minimum = obstacle_distances(samples, obs)
    # Recovering near a frame may begin inside the desired margin. Exclude only
    # the first few centimeters, which cannot be changed by a new route.
    deficit = np.maximum(0., margin - minimum[3:])
    return float(np.max(deficit, initial=0.)), float(deficit.sum() * 0.035)


def _can_retain_path(path: CubicHermiteSpline, arc: float, obs: dict, clearance: float) -> bool:
    """Check the remaining path against newly observed poles and gate apertures."""
    remaining_length = path.x[-1] - arc
    samples = path(np.linspace(arc, path.x[-1], max(4, int(remaining_length / 0.04) + 1)))
    if np.min(obstacle_distances(samples[2:], obs)) < clearance:
        return False
    first = 0
    progress = int(obs["n_gates_passed"])
    gates = obs["gate_sequence"][progress:]
    directions = obs["gate_sequence_direction"][progress:]
    for gate, direction in zip(gates, directions, strict=True):
        rotation = Rotation.from_quat(obs["gates_quat"][gate])
        local = rotation.inv().apply(samples[first:] - obs["gates_pos"][gate])
        along = direction * local[:, 0]
        crossings = np.flatnonzero((along[:-1] <= 0.) & (along[1:] > 0.))
        if not len(crossings):
            return False
        left = along[crossings]
        fraction = -left / (along[crossings + 1] - left)
        crossing = local[crossings, 1:] + fraction[:, None] * (
            local[crossings + 1, 1:] - local[crossings, 1:]
        )
        offset = np.max(np.abs(crossing), axis=1)
        # Ignore intersections outside the 0.45 m scoring aperture used by the racing environments.
        inside = np.flatnonzero(offset < 0.225)
        if not len(inside) or offset[inside[0]] > 0.095:
            return False
        first += crossings[inside[0]] + 1
    return True

def _detour_chord_is_clear(samples, obs):
    """Check a shortcut against the three-dimensional frame boxes."""
    if np.min(obstacle_distances(samples, obs)) < 0.128:
        return False

    unknown = ~obs["gates_visited"]
    if not np.any(unknown):
        return True
    centers = obs["gates_pos"][unknown]
    axes = Rotation.from_quat(obs["gates_quat"][unknown]).as_matrix()
    local = np.einsum("sgi,gij->sgj", samples[:, None] - centers, axes)
    # Keep more room for gate poses that still come from the nominal map.
    uncertainty = np.array([0.105, 0.105, 0.0735])
    delta = np.abs(local[:, :, None] - _GATE_BAR_CENTERS) - _GATE_BAR_SIZES - uncertainty
    distance = np.linalg.norm(np.maximum(delta, 0.), axis=-1)
    return bool(np.all(distance >= 0.093))


def _prune_spatial_detours(original, arguments, timing, obs):
    """Retest disjoint planar detours against the measured three-dimensional scene."""
    points, headings = arguments[:2]
    detour = np.array([heading is None for heading in headings])
    detour[[0, -1]] = False
    boundaries = np.flatnonzero(np.diff(np.r_[False, detour, False]))
    keep = np.ones(len(points), dtype=bool)
    for start, end in zip(boundaries[::2], boundaries[1::2], strict=True):
        origin = points[start - 1]
        chord = points[end] - origin
        if np.linalg.norm(chord) < 0.56:
            continue
        samples = origin + np.linspace(0., 1., 31)[:, None] * chord
        if _detour_chord_is_clear(samples, obs):
            keep[start:end] = False
    if keep.all():
        return original

    refined = timed_trajectory(
        points[keep], [heading for heading, retain in zip(headings, keep, strict=True) if retain],
        *arguments[2:], **timing,
    )
    if refined[2] >= original[2] - 0.13:
        return original
    path = refined[0]
    samples = path(np.linspace(0., path.x[-1], int(path.x[-1] / 0.035) + 3))[2:]
    if (np.any(samples < [-2.47, -1.47, -0.001]) or np.any(samples > [2.47, 1.47, 1.95])
            or not _can_retain_path(path, 0., obs, 0.108)):
        return original
    return refined


def plan_remaining_race(obs, settings):
    """Choose the shortest timed trajectory from bounded whole-route variations."""
    if hasattr(settings, "to_dict"):
        settings = settings.to_dict()

    progress = int(obs["n_gates_passed"])
    straight_acceleration = settings.get("straight_power", 0.) if progress > 0 else 0.
    open_limit = settings.get("open_lateral_acceleration", 0.)
    open_turn = (obs, open_limit) if progress > 0 and open_limit > settings["lateral_acceleration"] else None

    finish_runout = settings.get("finish_runout", 0.0)
    final_gate = int(obs["gate_sequence"][-1])
    extend_finish = (
        finish_runout > 0
        and progress == len(obs["gate_sequence"]) - 1
        and obs["gates_visited"][final_gate]
    )
    timing = {
        "acceleration_budget": settings.get("acceleration_budget"),
        "vector_speed_profile": settings.get("vector_speed_profile", False),
        "straight_acceleration": straight_acceleration,
        "open_turn": open_turn,
    }

    def build(alignment, short_alignment, offset, slope, speed_scale, minimal=False, clearance=None):
        points, headings = race_waypoints(
            obs, settings["clearance"] if clearance is None else clearance,
            alignment, offset, short_alignment, slope,
            settings.get("local_endpoint_clearance", False), minimal,
        )
        if extend_finish:
            # Keep the crossing direction and reserve space to brake after the gate.
            end = points[-1] + finish_runout * headings[-1]
            extension = np.linspace(points[-1], end, 9)
            if (
                np.all(extension >= [-2.40, -1.40, 0.10])
                and np.all(extension <= [2.40, 1.40, 1.90])
                and np.min(obstacle_distances(extension, obs)) >= 0.15
            ):
                points[-1] = end
        arguments = (
            points, headings, obs["vel"], settings["cruise_speed"] * speed_scale,
            settings["acceleration"] * speed_scale, settings["lateral_acceleration"],
        )
        return timed_trajectory(*arguments, **timing), arguments

    alignment = settings["alignment"]
    short_alignment = settings.get("short_alignment", 0.)
    aperture = settings.get("aperture_offset_limit", 0.)
    slope = settings.get("crossing_slope", 0.)
    best, best_arguments = build(alignment, short_alignment, aperture, slope, 1.)
    if (not settings.get("optimize_race_time", False)
            or progress < settings.get("optimizer_start_after", 0)):
        return best
    commit_distance = settings.get("optimizer_commit_distance", 0.0)
    target = obs["gate_sequence"][progress]
    if np.linalg.norm(obs["pos"] - obs["gates_pos"][target]) < commit_distance:
        return best

    margin = settings.get("optimizer_margin", 0.13)
    worst_allowed, total_allowed = clearance_deficit(best[0], obs, margin)
    if settings.get("compact_gate_search", False):
        candidates = (
            (0.24, 0.11, 0.035, 0.14, 1.0),
            (0.15, 0.09, 0.035, 0.18, 1.08),
            (0.07, 0.07, 0.025, 0.12, 1.12),
        )
    else:
        candidates = (
            (0.34, 0.15, 0.035, 0.16, 1.0),
            (0.30, 0.12, 0.055, 0.24, 1.0),
            (0.28, 0.10, 0.035, 0.10, 1.12),
        )
    candidates = [(*candidate, False, None) for candidate in candidates]
    if settings.get("minimal_gate_search", False):
        candidates.extend(((0.18, 0.08, 0.025, 0.12, 1.0, True, None),
                           (0.18, 0.08, 0.035, 0.28, 1.08, True, None)))
    if settings.get("narrow_graph_search", False):
        candidates.extend(((0.16, 0.09, 0.035, 0.18, 1.0, False, 0.32),
                           (0.10, 0.07, 0.025, 0.14, 1.10, False, 0.29)))
    preferred = None
    if settings.get("prefer_narrow_route", False) and settings.get("narrow_graph_search", False):
        preferred = candidates[-1]
        candidates = [preferred, *candidates[:-1]]
    for candidate in candidates:
        proposal, arguments = build(*candidate)
        if proposal[2] >= best[2] - 0.12:
            continue
        worst, total = clearance_deficit(proposal[0], obs, margin)
        if worst <= worst_allowed + 0.008 and total <= total_allowed + 0.002:
            # Fall back to the remaining candidates when the narrow route
            # cannot clear the same geometric and time checks.
            best, best_arguments = proposal, arguments
            if candidate == preferred:
                break
    if settings.get("spatial_detour_pruning", False) and np.any(obs["gates_visited"]):
        return _prune_spatial_detours(best, best_arguments, timing, obs)
    return best


def _finish_gate_plan(obs, settings):
    """Plan an approach from the required side, including a go-around after a missed pass."""
    gate = obs["gate_sequence"][-1]
    center = obs["gates_pos"][gate]
    rotation = Rotation.from_quat(obs["gates_quat"][gate])
    forward, side, _ = rotation.as_matrix().T
    forward *= obs["gate_sequence_direction"][-1]
    start = obs["pos"]
    along = (start - center) @ forward
    across = (start - center) @ side
    rotations = Rotation.from_quat(obs["gates_quat"])
    posts = rotations.apply(np.tile([0., 0.32, 0.], (len(rotations), 1)))
    objects = np.vstack((obs["obstacles_pos"], obs["gates_pos"] - posts, obs["gates_pos"] + posts))
    best, best_score = None, np.inf
    for depth in (0.30, 0.48, 0.72):
        entry = center - depth * forward
        for width in (0.0, 0.61, -0.61):
            if width == 0.0 and along > -0.12:
                continue
            waypoints = [start]
            headings = [None]
            if width and along > -depth:
                # Cross the gate plane beside the frame before returning to the entrance.
                before = center + 0.21 * forward + width * side
                after = entry + width * side
                detour = approach_route(start, before, objects, 0.23)
                waypoints.extend((*detour, before, after))
                headings.extend([None] * (len(detour) + 2))
            detour = approach_route(waypoints[-1], entry, objects, 0.23)
            waypoints.extend((*detour, entry, center, center + 0.52 * forward))
            headings.extend([None] * len(detour) + [forward] * 3)
            points = np.asarray(waypoints)
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.015]
            plan = timed_trajectory(
                points[keep], [h for h, ok in zip(headings, keep, strict=True) if ok],
                obs["vel"], settings["cruise_speed"], settings["acceleration"],
                settings["lateral_acceleration"], settings.get("acceleration_budget"),
                settings.get("vector_speed_profile", False),
            )
            path, _, duration = plan
            if not _can_retain_path(path, 0., obs, 0.095):
                continue
            score = duration + (0.04 if across * width < 0. else 0.)
            if score < best_score:
                best, best_score = plan, score
    return best


def _finish_retreat(obs):
    """Back through the center of a missed gate, then stop on the entrance side."""
    progress = int(obs["n_gates_passed"])
    gate = obs["gate_sequence"][progress]
    center = obs["gates_pos"][gate]
    forward = Rotation.from_quat(obs["gates_quat"][gate]).apply(
        [float(obs["gate_sequence_direction"][progress]), 0., 0.],
    )
    offset = obs["pos"] - center
    along = offset @ forward
    if not (-0.08 < along < 0.62) or np.linalg.norm(offset - along * forward) > 0.13:
        return None
    retreat = center - 0.41 * forward
    plan = timed_trajectory(
        np.vstack((obs["pos"], retreat)), [None, -forward], obs["vel"],
        1.1, 2.4, 2.5, vector_speed_profile=True,
    )
    path = plan[0]
    samples = path(np.linspace(0., path.x[-1], max(25, int(np.ceil(path.x[-1] / 0.025)) + 1)))
    if np.min(obstacle_distances(samples, obs)) < 0.075:
        return None
    return plan


def _finish_retry_frame(obs):
    """Express position relative to the final aperture and its scoring direction."""
    gate = obs["gate_sequence"][-1]
    center = obs["gates_pos"][gate]
    axis = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()[:, 0]
    axis *= obs["gate_sequence_direction"][-1]
    offset = obs["pos"] - center
    along = offset @ axis
    return center, axis, along, offset - along * axis


def _finish_retry_trajectory(obs):
    """Reverse and make a new pass in one trajectory after a centered miss."""
    center, axis, along, cross = _finish_retry_frame(obs)
    axial_speed = obs["vel"] @ axis
    if not (0.004 < along < 0.25 and -0.75 < axial_speed < -0.20
            and np.linalg.norm(cross) < 0.115):
        return None

    turn = center - 0.295 * axis
    end = center + 0.24 * axis
    zero = np.zeros(3)
    boundary = (
        (obs["pos"], obs["vel"], zero),
        (turn, zero, 2.8 * axis),
        (center, 0.72 * axis, 0.60 * axis),
        (end, zero, zero),
    )
    # The spline parameter is measured time. The identity clock keeps its
    # velocity and acceleration valid through the zero-speed reversal.
    for scale in (1., 1.13, 1.28):
        times = np.r_[0., np.cumsum(scale * np.array([0.64, 0.57, 0.43]))]
        path = BPoly.from_derivatives(times, boundary)
        grid = np.linspace(0., times[-1], 78)
        position = path(grid)
        acceleration = path(grid, 2)
        if (np.max(np.linalg.norm(acceleration[:, :2], axis=1)) > 4.55
                or np.max(np.abs(acceleration[:, 2])) > 2.6):
            continue
        if (np.min(obstacle_distances(position, obs)) < 0.081
                or np.any(position < [-2.47, -1.47, 0.07])
                or np.any(position > [2.47, 1.47, 1.95])):
            return None
        return path, PchipInterpolator([0., times[-1]], [0., times[-1]]), times[-1]
    return None


def _gate_exit_window(path, clock, duration, center, normal):
    """Find a centered crossing and a short runout on the planned path."""
    count = max(20, int(path.x[-1] / 0.028) + 1)
    grid = np.unique(np.r_[np.linspace(0., path.x[-1], count), path.x])
    relative = path(grid) - center
    signed = relative @ normal
    crossing = np.flatnonzero(
        (signed[:-1] <= 0.) & (signed[1:] > 0.)
        & (np.linalg.norm(relative[:-1], axis=1) < 0.17)
    )
    if not len(crossing):
        return None
    index = crossing[0]

    def normal_distance(arc):
        return (path(arc) - center) @ normal

    left, right = grid[index:index + 2]
    begin, finish = normal_distance(left), normal_distance(right)
    if abs(begin) < 1e-8:
        root = left
    elif abs(finish) < 1e-8:
        root = right
    elif begin < 0. < finish:
        root = brentq(normal_distance, left, right)
    else:
        return None
    if np.linalg.norm(path(root) - center) > 0.10:
        return None

    after = signed[index + 1:]
    lateral = relative[index + 1:] - after[:, None] * normal
    exit_points = np.flatnonzero(
        (after > 0.09) & (after < 0.22) & (np.linalg.norm(lateral, axis=1) < 0.105)
    )
    if not len(exit_points):
        return None
    exit_arc = grid[index + 1 + exit_points[0]]
    crossing_time = brentq(lambda when: clock(when) - root, 0., duration)
    exit_time = brentq(lambda when: clock(when) - exit_arc, crossing_time, duration)
    return crossing_time, exit_time


class Level3MPC(AttitudeMPC):
    """Plan ahead across gates and track the current horizon with the existing MPC."""

    racing_tracking = True

    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self._obstacle_planes_active = None
        self.begin_race()

    def begin_race(self):
        """Reset race-local controller state and cache the race settings."""
        self._acados_ocp_solver.reset()
        self._settings = self._config.controller.to_dict()
        self._tick = 0
        self._finished = False
        self._geometry = None
        self._retreating = False
        self._reference_time = 0.0
        self._last_command = np.array([0., 0., 0., self.drone_params["mass"] * 9.81])
        self.solve_failures = 0
        self._gate_hold_path = None
        self._gate_hold_seen_path = None
        self._gate_hold_until = 0
        self._gate_hold_cap = 0.

    def _update_gate_exit_hold(self, obs, progress):
        """Keep the preview on the fourth gate until its pass is registered."""
        if progress != 3 or len(obs["gate_sequence"]) != 5 or self._geometry is None:
            self._gate_hold_path = None
            return
        if self._tick >= self._gate_hold_until:
            self._gate_hold_path = None

        gate = obs["gate_sequence"][progress]
        if self._gate_hold_seen_path is self._path or not obs["gates_visited"][gate]:
            return
        normal = Rotation.from_quat(obs["gates_quat"][gate]).apply([1., 0., 0.])
        normal *= obs["gate_sequence_direction"][progress]
        center = obs["gates_pos"][gate]
        offset = obs["pos"] - center
        signed = offset @ normal
        if not (-0.34 < signed < -0.075 and np.linalg.norm(offset - signed * normal) < 0.21):
            return
        now = self._reference_time
        if now >= self._duration:
            return
        reference = self._path(self._time_map(now)) - center
        if not (-0.10 < reference @ normal < 0.19):
            return

        window = _gate_exit_window(self._path, self._time_map, self._duration, center, normal)
        self._gate_hold_seen_path = self._path
        if window is not None and -0.26 < window[0] - now < 0.24:
            self._gate_hold_path = self._path
            self._gate_hold_until = self._tick + round(1.1 / self._dt)
            self._gate_hold_cap = window[1]

    def _finish_is_imminent(self, obs):
        """Preserve forward motion while the drone is still approaching the aperture."""
        center, axis, along, cross = _finish_retry_frame(obs)
        axial_speed = obs["vel"] @ axis
        if not (-0.086 < along < 0. and axial_speed >= 0.22 and np.linalg.norm(cross) <= 0.10):
            return False

        arrival = -along / axial_speed
        lateral_velocity = obs["vel"] - axial_speed * axis
        if arrival > 0.20 or np.linalg.norm(cross + arrival * lateral_velocity) > 0.11:
            return False
        now = min(self._reference_time, self._duration)
        horizon = np.linspace(now, min(self._duration, now + 0.32), 12)
        relative = self._path(self._time_map(horizon)) - center
        safe_preview = (np.linalg.norm(relative, axis=1) < 0.13) & (relative @ axis > 0.02)
        return bool(np.any(safe_preview))

    def compute_control(self, obs, info=None):
        progress = int(obs["n_gates_passed"])
        if progress >= len(obs["gate_sequence"]):
            return self._last_command
        hold_gate_exit = self._settings.get("hold_gate_exit", False)
        if hold_gate_exit:
            self._update_gate_exit_hold(obs, progress)
        geometry = np.concatenate((
            obs["gates_pos"].ravel(), obs["gates_quat"].ravel(), obs["obstacles_pos"].ravel(),
        ))
        settings = self._settings
        changed = self._geometry is None or np.max(np.abs(geometry - self._geometry)) > 0.035
        clearance = settings.get("retain_clearance")
        if (changed and clearance is not None and self._geometry is not None
                and self._reference_time < self._duration):
            arc = self._time_map(self._reference_time)
            tracking_error = np.linalg.norm(obs["pos"] - self._path(arc))
            if tracking_error < 0.16 and _can_retain_path(self._path, arc, obs, clearance):
                self._geometry = geometry
                changed = False
        expired = self._geometry is not None and self._reference_time > self._duration + 0.35
        finish = progress == len(obs["gate_sequence"]) - 1
        recover_finish = finish and settings.get("recover_finish_gate", False)
        retreat = None
        if recover_finish and not self._retreating:
            gate = obs["gate_sequence"][progress]
            distance = np.linalg.norm(obs["pos"] - obs["gates_pos"][gate])
            if distance < 0.70 and (expired or np.linalg.norm(obs["vel"]) < 0.48):
                retreat = _finish_retreat(obs)
                if (retreat is not None and obs["gates_visited"][gate]
                        and settings.get("continuous_finish_retry", False)):
                    if not expired and self._finish_is_imminent(obs):
                        retreat = None
                    else:
                        retry = _finish_retry_trajectory(obs)
                        if retry is not None:
                            retreat = retry
        replanned = changed or expired or retreat is not None
        if self._retreating and not expired and changed:
            # Recheck the active retreat if new geometry arrives mid-maneuver.
            arc = self._time_map(min(self._reference_time, self._duration))
            samples = self._path(np.linspace(arc, self._path.x[-1], 25))
            if np.min(obstacle_distances(samples, obs)) >= 0.075:
                self._geometry = geometry
                replanned = False
        if replanned:
            initial_velocity = obs["vel"]
            if settings.get("blend_replan_velocity", False) and self._geometry is not None:
                old_time = min(self._reference_time, self._duration)
                old_arc = self._time_map(old_time)
                old_position = self._path(old_arc)
                old_velocity = self._path(old_arc, 1) * self._time_map(old_time, 1)
                if (np.linalg.norm(old_position - obs["pos"]) < 0.15
                        and initial_velocity @ old_velocity > 0
                        and np.linalg.norm(old_velocity - initial_velocity) < 0.9):
                    initial_velocity = 0.65 * initial_velocity + 0.35 * old_velocity
            planning_obs = dict(obs, vel=initial_velocity)
            plan = retreat
            if plan is None and recover_finish and expired:
                plan = _finish_gate_plan(planning_obs, settings)
            self._retreating = retreat is not None
            if plan is None:
                plan = plan_remaining_race(planning_obs, settings)
            self._path, self._time_map, self._duration = plan
            self._geometry = geometry
            self._reference_time = 0.0

        current_arc = self._time_map(min(self._reference_time, self._duration))
        residual = obs["pos"] - self._path(current_arc)
        predictive = settings.get("predictive_race_tracking", False) and progress >= settings.get(
            "predictive_start_after", 0
        )
        phase_gain = 0. if predictive else settings.get("phase_correction", 0.)
        if phase_gain and not replanned and not self._retreating:
            tangent = self._path(current_arc, 1)
            tangent /= max(np.linalg.norm(tangent), 0.1)
            along = residual @ tangent
            lateral = np.linalg.norm(residual - along * tangent)
            expected = np.linalg.norm(self._path(current_arc, 1) * self._time_map(
                min(self._reference_time, self._duration), 1,
            ))
            if lateral < 0.12 and expected > 0.65 and obs["vel"] @ tangent > 0.35:
                max_lag = settings.get("phase_lag_limit", 0.012)
                shift = np.clip(phase_gain * along / expected, -max_lag, 0.035)
                self._reference_time = max(0., self._reference_time + shift)
                current_arc = self._time_map(min(self._reference_time, self._duration))
                residual = obs["pos"] - self._path(current_arc)
        error = np.linalg.norm(residual)
        clock_rate = np.clip(1.0 - 2.5 * max(0.0, error - 0.10), 0.4, 1.0)
        if predictive and not self._retreating:
            gate = obs["gate_sequence"][progress]
            gate_distance = np.linalg.norm(obs["pos"] - obs["gates_pos"][gate])
            pole_distances = np.linalg.norm(obs["obstacles_pos"][:, :2] - obs["pos"][:2], axis=1)
            if gate_distance > 0.70 and np.all(pole_distances > 0.55):
                tangent = self._path(current_arc, 1)
                tangent /= max(np.linalg.norm(tangent), 0.1)
                along = residual @ tangent
                cross_track = np.linalg.norm(residual - along * tangent)
                # Let the MPC recover small along-track lags in open sections.
                lag_allowance = settings.get("open_tracking_lag", 0.16)
                transition = settings.get("open_tracking_transition", 0.0)
                if transition > 0.0:
                    clearance = min(gate_distance - 0.70, pole_distances.min(initial=np.inf) - 0.55)
                    blend = min(clearance / transition, 1.0)
                    blend = blend * blend * (3.0 - 2.0 * blend)
                    recovery_error = np.hypot(cross_track, min(along + lag_allowance, 0.0))
                    error += blend * (recovery_error - error)
                else:
                    error = np.hypot(cross_track, min(along + lag_allowance, 0.0))
                clock_rate = np.clip(1.0 - 2.5 * max(0.0, error - 0.10), 0.4, 1.0)
        times = self._reference_time + np.arange(self._N + 1) * self._prediction_dt * clock_rate
        self._reference_time += self._dt * clock_rate
        holding = hold_gate_exit and self._gate_hold_path is self._path
        if holding:
            cap = self._gate_hold_cap
            self._reference_time = min(self._reference_time, cap)
            capped = times >= cap
            times = np.minimum(times, cap)
        position, velocity, acceleration = self.tracking_reference(times, clock_rate)
        if holding:
            velocity[capped] = 0.
            acceleration[capped] = 0.
        action = self.track_reference(obs, position, velocity, acceleration)
        if self.solver_status not in (0, 2) or not np.isfinite(action).all():
            self.solve_failures += 1
            return self._last_command
        obs, position, velocity, acceleration = self._feedback_reference(
            obs, position, velocity, acceleration
        )
        # Blend the MPC thrust with vertical feedback using the tracking attitude.
        # This compensates for thrust/model mismatch under the randomized mass.
        if settings.get("altitude_feedback", False):
            mass = self.drone_params["mass"]
            up = Rotation.from_quat(obs["quat"]).as_matrix()[2, 2]
            az = acceleration[0, 2] + 8 * (position[0, 2] - obs["pos"][2])
            az += 4 * (velocity[0, 2] - obs["vel"][2])
            thrust = mass * (9.81 + az) / max(up, 0.6)
            action[3] = np.clip(0.55 * action[3] + 0.45 * thrust,
                                4 * self.drone_params["thrust_min"],
                                4 * self.drone_params["thrust_max"])
        if settings.get("aperture_feedback", False) and not self._retreating:
            self.center_on_gate(obs, progress, action, predictive)
        self._last_command = action
        return action

    def _feedback_reference(
        self, obs: dict, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Use the same observation and reference as the trajectory tracker."""
        return obs, position, velocity, acceleration

    def track_reference(self, obs, position, velocity, acceleration=None, yaw=0.0):
        if self._soft_obstacle_clearance:
            self._set_obstacle_clearance(obs, position[1:-1])
        return super().track_reference(obs, position, velocity, acceleration, yaw=yaw)

    def _set_obstacle_clearance(self, obs, position):
        """Add supporting planes around visible obstacles at the prediction nodes."""
        forecast = None
        solver = self._acados_ocp_solver
        if self._tick and getattr(self, "solver_status", None) in (0, 2):
            states = solver.get_flat("x").reshape(self._N + 1, self._nx)[:, :3]
            fraction = min(1., self._dt / self._prediction_dt)
            aligned = states[:-1] + fraction * np.diff(states, axis=0)
            if np.linalg.norm(aligned[0] - obs["pos"]) < .19:
                forecast = aligned[1:]
        gates = obs["gates_visited"]
        poles = obs["obstacles_visited"]
        rotations = (Rotation.from_quat(obs["gates_quat"][gates]).as_matrix()
                     if gates.any() else np.empty((0, 3, 3)))
        planes = build_clearance_planes(
            position, obs["gates_pos"][gates], rotations, obs["obstacles_pos"][poles],
            forecast=forecast,
        )
        self._obstacle_planes_active = set_clearance_planes(
            solver, planes, self._nx, self._obstacle_planes_active,
        )

    def tracking_reference(self, times, clock_rate):
        """Sample the timed path and its derivatives at MPC prediction stages."""
        clipped = np.minimum(times, self._duration)
        arc = self._time_map(clipped)
        rate = self._time_map(clipped, 1) * clock_rate
        tangent = self._path(arc, 1)
        velocity = tangent * rate[:, None]
        acceleration = (self._path(arc, 2) * rate[:, None] ** 2
                        + tangent * self._time_map(clipped, 2)[:, None] * clock_rate ** 2)
        velocity[times >= self._duration] = 0.
        acceleration[times >= self._duration] = 0.
        # Smooth interpolation can spike between speed samples. Limit feedforward tilt.
        acceleration = np.clip(acceleration, -4., 4.)
        position = self._path(arc)
        return position, velocity, acceleration

    def center_on_gate(self, obs, progress, action, predictive=False):
        """Damp cross-track motion near the target aperture."""
        gate = obs["gate_sequence"][progress]
        axis = Rotation.from_quat(obs["gates_quat"][gate]).apply([1., 0., 0.])
        axis *= obs["gate_sequence_direction"][progress]
        displacement = obs["pos"] - obs["gates_pos"][gate]
        along = displacement @ axis
        lateral = displacement - along * axis
        weight = np.clip((0.52 - abs(along)) / 0.30, 0., 1.)
        weight *= np.clip((0.36 - np.linalg.norm(lateral)) / 0.18, 0., 1.)
        if weight <= 0:
            return
        velocity = obs["vel"]
        lateral_velocity = velocity - (velocity @ axis) * axis
        correction = -weight * (9.0 * lateral + 3.0 * lateral_velocity)
        if predictive and obs["gates_visited"][gate]:
            # Correct the predicted intersection without damping motion
            # that is already carrying the drone through the aperture.
            forward_speed = velocity @ axis
            if along < 0.0 and forward_speed > 0.6:
                crossing_time = min(-along / forward_speed, 0.45)
                crossing_error = lateral + crossing_time * lateral_velocity
                correction = -weight * (11.0 * crossing_error + lateral_velocity)
        action[0] -= np.clip(correction[1] / 9.81, -0.17, 0.17)
        action[1] += np.clip(correction[0] / 9.81, -0.17, 0.17)
        action[3] += self.drone_params["mass"] * np.clip(correction[2], -1.8, 1.8)
        tilt_limit = self._ocp.constraints.ubu[0]
        np.clip(action[:2], -tilt_limit, tilt_limit, out=action[:2])
        action[3] = np.clip(action[3], 4 * self.drone_params["thrust_min"],
                           4 * self.drone_params["thrust_max"])
