from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from mathutils import Matrix, Vector

from infinigen.core.constraints.constraint_language import util as iu
from infinigen.core.constraints.example_solver import clusters
from infinigen.core.constraints.example_solver.geometry import dof
from infinigen.core.constraints.example_solver.moves.addition import Resample
from infinigen.core.constraints.example_solver.state_def import State

from . import moves
from .reassignment import pose_backup, restore_pose_backup


def _rotate_obj_about_point(obj, pivot: np.ndarray, axis: np.ndarray, angle: float, scene):
    pivot_vec = Vector(pivot.tolist() if isinstance(pivot, np.ndarray) else pivot)
    axis_vec = Vector(axis.tolist() if isinstance(axis, np.ndarray) else axis)
    if axis_vec.length == 0:
        raise ValueError("Rotation axis has zero length")
    axis_vec.normalize()

    to_origin = Matrix.Translation(-pivot_vec)
    back = Matrix.Translation(pivot_vec)
    rot = Matrix.Rotation(angle, 4, axis_vec)

    obj.matrix_world = back @ rot @ to_origin @ obj.matrix_world
    iu.sync_trimesh(scene, obj.name)


@dataclass
class ClusterTranslateMove(moves.Move):
    translation: np.ndarray
    cluster_parent: str
    children: Tuple[str, ...] = tuple()

    _backup_poses: Dict[str, dict] | None = None

    def __repr__(self):
        norm = np.linalg.norm(self.translation)
        return (
            f"{self.__class__.__name__}(parent={self.cluster_parent}, "
            f"count={len(self.names)}, dist={norm:.2e})"
        )

    def apply(self, state: State):
        self._backup_poses = {}
        for name in self.names:
            obj_state = state.objs.get(name)
            if obj_state is None or obj_state.obj is None:
                return False
            self._backup_poses[name] = pose_backup(obj_state, dof=False)
            iu.translate(state.trimesh_scene, obj_state.obj.name, self.translation)

        if not clusters.cluster_is_collision_free(state, self.names):
            self._restore(state)
            return False

        return True

    def revert(self, state: State):
        self._restore(state)

    def _restore(self, state: State):
        if not self._backup_poses:
            return
        for name, pose in self._backup_poses.items():
            restore_pose_backup(state, name, pose)


@dataclass
class ClusterRotateMove(moves.Move):
    axis: np.ndarray
    angle: float
    cluster_parent: str
    children: Tuple[str, ...] = tuple()

    _backup_poses: Dict[str, dict] | None = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(parent={self.cluster_parent}, "
            f"count={len(self.names)}, angle={self.angle:.2e})"
        )

    def apply(self, state: State):
        parent_state = state.objs.get(self.cluster_parent)
        if parent_state is None or parent_state.obj is None:
            return False

        pivot = clusters.object_center(parent_state.obj)
        axis = np.array(self.axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return False
        axis = axis / norm

        self._backup_poses = {}
        for name in self.names:
            obj_state = state.objs.get(name)
            if obj_state is None or obj_state.obj is None:
                return False
            self._backup_poses[name] = pose_backup(obj_state, dof=False)
            _rotate_obj_about_point(obj_state.obj, pivot, axis, self.angle, state.trimesh_scene)

        if not clusters.cluster_is_collision_free(state, self.names):
            self._restore(state)
            return False

        return True

    def revert(self, state: State):
        self._restore(state)

    def _restore(self, state: State):
        if not self._backup_poses:
            return
        for name, pose in self._backup_poses.items():
            restore_pose_backup(state, name, pose)


@dataclass
class ClusterResampleMove(moves.Move):
    cluster_parent: str
    children: Tuple[str, ...] = tuple()

    _parent_resample: Resample | None = None
    _child_backups: Dict[str, dict] | None = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(parent={self.cluster_parent}, "
            f"children={len(self.children)})"
        )

    def apply(self, state: State):
        parent_state = state.objs.get(self.cluster_parent)
        if parent_state is None or parent_state.generator is None:
            return False

        self._parent_resample = Resample(names=[self.cluster_parent])
        if not self._parent_resample.apply(state):
            self._parent_resample = None
            return False

        self._child_backups = {}
        for child in self.children:
            if child not in state.objs:
                continue
            obj_state = state.objs[child]
            self._child_backups[child] = pose_backup(obj_state)
            if not dof.try_apply_relation_constraints(state, child):
                self._restore_children(state)
                self._parent_resample.revert(state)
                self._parent_resample = None
                return False

        if not clusters.cluster_is_collision_free(state, self.names):
            self._restore_children(state)
            if self._parent_resample is not None:
                self._parent_resample.revert(state)
                self._parent_resample = None
            return False

        return True

    def revert(self, state: State):
        self._restore_children(state)
        if self._parent_resample is not None:
            self._parent_resample.revert(state)

    def accept(self, state: State):
        if self._parent_resample is not None:
            self._parent_resample.accept(state)

    def _restore_children(self, state: State):
        if not self._child_backups:
            return
        for name, pose in self._child_backups.items():
            restore_pose_backup(state, name, pose)
