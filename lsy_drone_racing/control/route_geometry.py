"""Shared obstacle detour geometry for the race controllers."""

from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def _edge_geometry(starts, ends):
    dx = ends[None, :, 0] - starts[:, None, 0]
    dy = ends[None, :, 1] - starts[:, None, 1]
    length_sq = dx ** 2 + dy ** 2
    return dx, dy, length_sq, np.maximum(length_sq, 1e-10)


def _squared_distances_from_edges(starts, edge, centers):
    dx, dy, _, denominator = edge
    ox = centers[:, None, None, 0] - starts[None, :, None, 0]
    oy = centers[:, None, None, 1] - starts[None, :, None, 1]
    fraction = np.clip((ox * dx + oy * dy) / denominator, 0., 1.)
    return (ox - fraction * dx) ** 2 + (oy - fraction * dy) ** 2


@lru_cache(maxsize=8)
def _obstacle_graph(centers_key, clearance):
    """Store edge geometry between detour nodes, shared by all route legs."""
    centers = np.asarray(centers_key)
    angles = np.arange(12) * (2 * np.pi / 12)
    ring = 1.08 * clearance * np.column_stack((np.cos(angles), np.sin(angles)))
    nodes = (centers[:, None] + ring).reshape(-1, 2)
    edge = _edge_geometry(nodes, nodes)
    # Reuse two edge-sized work arrays to avoid allocating several full
    # obstacle-by-edge tensors on each graph cache miss.
    dx, dy, length_sq, denominator = edge
    distances = np.empty((len(centers), len(nodes), len(nodes)), dtype=nodes.dtype)
    projection = np.empty_like(dx)
    component = np.empty_like(dy)
    for center, distance in zip(centers, distances, strict=True):
        ox = (center[0] - nodes[:, 0])[:, None]
        oy = (center[1] - nodes[:, 1])[:, None]
        np.multiply(ox, dx, out=projection)
        np.multiply(oy, dy, out=component)
        projection += component
        projection /= denominator
        np.clip(projection, 0.0, 1.0, out=projection)
        np.multiply(projection, dx, out=distance)
        np.subtract(ox, distance, out=distance)
        np.multiply(projection, dy, out=component)
        np.subtract(oy, component, out=component)
        np.square(distance, out=distance)
        np.square(component, out=component)
        distance += component
    collisions = distances < max(clearance - 1e-6, 0.) ** 2
    return nodes, np.sqrt(length_sq), distances, collisions, collisions.sum(axis=0)


def approach_route(
    start, end, obstacles, clearance=0.27, *, local_endpoint_clearance=False,
):
    """Find planar waypoints around poles using a small visibility graph."""
    centers = np.asarray(obstacles)[:, :2]
    if not len(centers):
        return []
    endpoints = np.vstack((start[:2], end[:2]))
    direct = _edge_geometry(endpoints[:1], endpoints[1:])
    endpoint_radii = np.minimum(clearance, np.linalg.norm(endpoints[:, None] - centers, axis=-1))
    direct_clear = np.all(
        _squared_distances_from_edges(endpoints[:1], direct, centers)[:, 0, 0]
        >= np.maximum(endpoint_radii.min(axis=0) - 1e-6, 0.) ** 2
    )
    if direct_clear:
        return []
    key = tuple(map(tuple, centers))
    inner, inner_costs, inner_distances, inner_collisions, collision_count = _obstacle_graph(
        key, clearance,
    )
    nodes = np.vstack((start[:2], end[:2], inner))
    # Compute only the edges touching the dynamic start and destination.
    from_endpoints = _edge_geometry(nodes[:2], nodes)
    costs = np.empty((len(nodes), len(nodes)))
    costs[2:, 2:] = inner_costs
    costs[:2, :] = np.sqrt(from_endpoints[2])
    costs[2:, :2] = costs[:2, 2:].T
    if local_endpoint_clearance:
        endpoint_limits = endpoint_radii.T
        destination_limits = np.column_stack((
            endpoint_limits, np.full((len(centers), len(inner)), clearance),
        ))
        from_limits = np.minimum(endpoint_limits[:, :, None], destination_limits[:, None, :])
        inner_blocked = collision_count > 0
    else:
        shared = endpoint_radii.min(axis=0)
        from_limits = shared[:, None, None]
        reduced = shared < clearance
        if reduced.any():
            relieved = inner_collisions[reduced] & (
                inner_distances[reduced] >= np.maximum(shared[reduced, None, None] - 1e-6, 0.) ** 2
            )
            inner_blocked = collision_count > relieved.sum(axis=0)
        else:
            inner_blocked = collision_count > 0
    outward = _squared_distances_from_edges(nodes[:2], from_endpoints, centers)
    blocked = np.empty_like(costs, dtype=bool)
    blocked[:2, :] = np.any(outward < np.maximum(from_limits - 1e-6, 0.) ** 2, axis=0)
    blocked[2:, :2] = blocked[:2, 2:].T
    blocked[2:, 2:] = inner_blocked
    np.fill_diagonal(blocked, True)
    # Nonzero returns entries in row-major order, so build CSR directly.
    rows, columns = np.nonzero(~blocked)
    offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=len(nodes)))]
    graph = csr_matrix((costs[rows, columns], columns, offsets), shape=costs.shape)
    _, parents = dijkstra(graph, directed=True, indices=0, return_predecessors=True)
    if parents[1] < 0:
        if local_endpoint_clearance:
            return approach_route(start, end, obstacles, clearance)
        return []
    path = []
    node = parents[1]
    while node:
        path.append(nodes[node])
        node = parents[node]
    path.reverse()
    chain = np.vstack((start[:2], path, end[:2]))
    lengths = np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))
    heights = start[2] + (end[2] - start[2]) * lengths[:-1] / lengths[-1]
    return np.column_stack((path, heights)).tolist()
