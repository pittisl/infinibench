from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl, reasoning as r
from infinigen.core.constraints.evaluator.domain_contains import domain_contains
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.util import blender as butil

# In InfiniBench, clusters are defined dynamically from the scene graph:
# - The cluster root is the highest-level object (eg. a table) that other objects are stably placed on.
# - Child objects are any objects that maintain a StableAgainst relationship with the root.
# This module keeps the definitions centralized so the optimizer can rebuild clusters from the latest state.

CLUSTER_RELATIONS = (cl.StableAgainst,)


@dataclass(frozen=True)
class MovableCluster:
    parent: str
    children: tuple[str, ...]

    def members(self) -> tuple[str, ...]:
        return (self.parent, *self.children)

    def bounding_box(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        return cluster_bounding_box(state, self.members())

    def size(self) -> int:
        return 1 + len(self.children)


def identify_movable_clusters(
    state: State,
    filter_domain: r.Domain | None = None,
    min_children: int = 1,
) -> list[MovableCluster]:
    """Group related objects (table -> chairs, desk -> monitors, etc.) into movable clusters."""

    def selectable(obj_state) -> bool:
        if obj_state is None or obj_state.obj is None:
            return False
        if not obj_state.active:
            return False
        if t.Semantics.Room in obj_state.tags:
            return False
        return True

    parent_children: dict[str, set[str]] = defaultdict(set)

    for child_name, obj_state in state.objs.items():
        if not selectable(obj_state):
            continue
        for rel_state in obj_state.relations:
            if rel_state.target_name not in state.objs:
                continue
            if not isinstance(rel_state.relation, CLUSTER_RELATIONS):
                continue
            parent_state = state.objs[rel_state.target_name]
            if not selectable(parent_state):
                continue
            if parent_state is obj_state:
                continue
            if t.Semantics.NoChildren in parent_state.tags:
                continue
            parent_children[rel_state.target_name].add(child_name)

    clusters: list[MovableCluster] = []
    for parent, children in parent_children.items():
        if len(children) < min_children:
            continue
        cluster = MovableCluster(parent=parent, children=tuple(sorted(children)))
        if filter_domain is not None:
            if not any(
                domain_contains(filter_domain, state, state.objs[name])
                for name in cluster.members()
            ):
                continue
        clusters.append(cluster)

    return clusters


def object_aabb(obj) -> tuple[np.ndarray, np.ndarray]:
    corners = butil.apply_matrix_world(obj, np.array(obj.bound_box, dtype=float))
    return corners.min(axis=0), corners.max(axis=0)


def object_center(obj) -> np.ndarray:
    mn, mx = object_aabb(obj)
    return (mn + mx) * 0.5


def cluster_bounding_box(
    state: State,
    members: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for name in members:
        if name not in state.objs:
            continue
        obj_state = state.objs[name]
        if obj_state.obj is None:
            continue
        mn, mx = object_aabb(obj_state.obj)
        mins.append(mn)
        maxs.append(mx)

    if not mins:
        zeros = np.zeros(3)
        return zeros, zeros

    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def boxes_overlap(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
    padding: float = 0.0,
) -> bool:
    for axis in range(3):
        if (a_max[axis] + padding) <= (b_min[axis] - padding):
            return False
        if (b_max[axis] + padding) <= (a_min[axis] - padding):
            return False
    return True


def cluster_is_collision_free(
    state: State,
    members: Iterable[str],
    padding: float = 0.0,
) -> bool:
    members = tuple(members)
    cluster_min, cluster_max = cluster_bounding_box(state, members)

    for name, obj_state in state.objs.items():
        if name in members:
            continue
        if not obj_state.active or obj_state.obj is None:
            continue
        if t.Semantics.NoCollision in obj_state.tags:
            continue

        other_min, other_max = object_aabb(obj_state.obj)
        if boxes_overlap(cluster_min, cluster_max, other_min, other_max, padding=padding):
            return False

    return True
