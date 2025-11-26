from __future__ import annotations

import logging
from typing import Callable, Iterator

import numpy as np

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.example_solver import clusters, state_def
from infinigen.core.constraints.example_solver.moves.cluster import (
    ClusterResampleMove,
    ClusterRotateMove,
    ClusterTranslateMove,
)
from infinigen.core.constraints.example_solver.propose_continous import (
    ANGLE_STEP_SIZE,
    ROT_MULT,
    ROT_MIN,
)

logger = logging.getLogger(__name__)

CLUSTER_TRANS_MULT = 4.0
CLUSTER_TRANS_MIN = 0.05


def _cluster_candidates(
    state: state_def.State,
    filter_domain: r.Domain | None,
    predicate: Callable[[clusters.MovableCluster, state_def.ObjectState], bool] = None,
) -> list[clusters.MovableCluster]:
    all_clusters = clusters.identify_movable_clusters(state, filter_domain)
    candidates = []
    for cluster in all_clusters:
        parent_state = state.objs.get(cluster.parent)
        if parent_state is None:
            continue
        if predicate is not None and not predicate(cluster, parent_state):
            continue
        candidates.append(cluster)
    return candidates


def _sample_translation(os: state_def.ObjectState, temperature: float):
    if os.dof_matrix_translation is None:
        return None
    var = max(CLUSTER_TRANS_MIN, CLUSTER_TRANS_MULT * temperature)
    random_vector = np.random.normal(0, var, size=3)
    translation = os.dof_matrix_translation @ random_vector
    if np.linalg.norm(translation) < 1e-4:
        return None
    return translation


def _sample_rotation_angle(temperature: float):
    var = max(ROT_MIN, ROT_MULT * temperature)
    random_angle = np.random.normal(0, var)
    ang = random_angle / ANGLE_STEP_SIZE
    ang = np.ceil(ang) if ang > 0 else np.floor(ang)
    return ang * ANGLE_STEP_SIZE


def _empty_generator():
    return iter(())


def propose_cluster_translate(
    consgraph: cl.Node,
    state: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
) -> Iterator[ClusterTranslateMove]:
    del consgraph  # unused, kept for interface parity

    candidates = _cluster_candidates(
        state,
        filter_domain,
        predicate=lambda _cluster, parent: parent.dof_matrix_translation is not None,
    )
    if not candidates:
        logger.debug("No movable clusters available for cluster translate move")
        return _empty_generator()

    weights = np.array([cluster.size() for cluster in candidates], dtype=float)
    probs = weights / weights.sum()

    def _generator():
        while True:
            idx = np.random.choice(len(candidates), p=probs)
            cluster = candidates[idx]
            parent_state = state.objs[cluster.parent]
            translation = _sample_translation(parent_state, temperature)
            if translation is None:
                continue

            yield ClusterTranslateMove(
                names=list(cluster.members()),
                translation=translation,
                cluster_parent=cluster.parent,
                children=cluster.children,
            )

    return _generator()


def propose_cluster_rotate(
    consgraph: cl.Node,
    state: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
) -> Iterator[ClusterRotateMove]:
    del consgraph

    def predicate(_cluster, parent: state_def.ObjectState):
        if parent.dof_rotation_axis is None:
            return False
        if t.Semantics.NoRotation in parent.tags:
            return False
        return True

    candidates = _cluster_candidates(state, filter_domain, predicate=predicate)
    if not candidates:
        logger.debug("No movable clusters available for cluster rotate move")
        return _empty_generator()

    weights = np.array([cluster.size() for cluster in candidates], dtype=float)
    probs = weights / weights.sum()

    def _generator():
        while True:
            idx = np.random.choice(len(candidates), p=probs)
            cluster = candidates[idx]
            parent_state = state.objs[cluster.parent]
            angle = _sample_rotation_angle(temperature)
            if abs(angle) < 1e-6:
                continue
            yield ClusterRotateMove(
                names=list(cluster.members()),
                axis=np.array(parent_state.dof_rotation_axis),
                angle=angle,
                cluster_parent=cluster.parent,
                children=cluster.children,
            )

    return _generator()


def propose_cluster_resample(
    consgraph: cl.Node,
    state: state_def.State,
    filter_domain: r.Domain,
    temperature: float,
) -> Iterator[ClusterResampleMove]:
    del consgraph
    del temperature

    candidates = _cluster_candidates(
        state,
        filter_domain,
        predicate=lambda _cluster, parent: parent.generator is not None,
    )
    if not candidates:
        logger.debug("No movable clusters available for cluster resample move")
        return _empty_generator()

    weights = np.array([cluster.size() for cluster in candidates], dtype=float)
    probs = weights / weights.sum()

    def _generator():
        while True:
            idx = np.random.choice(len(candidates), p=probs)
            cluster = candidates[idx]
            yield ClusterResampleMove(
                names=list(cluster.members()),
                cluster_parent=cluster.parent,
                children=cluster.children,
            )

    return _generator()
