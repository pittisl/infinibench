"""Camera trajectory optimization for InfiniBench scenes.

The implementation follows the frontier-based procedure described in the paper:
1. Select the closest unvisited target object.
2. Sample and score candidate viewpoints around the target.
3. Use Dijkstra on a 2D navigation graph (constant camera height) to find a
   collision-free path towards the best-scoring viewpoint.
4. Iterate until every target is observed, returning the resulting trajectory.

The optimizer expects to run inside Blender's Python interpreter.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import bpy
import networkx as nx
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


@dataclass
class CameraPose:
    position: tuple[float, float, float]
    rotation_euler: tuple[float, float, float]


@dataclass
class TrajectoryResult:
    poses: list[CameraPose]

    def to_json(self) -> list[dict]:
        return [
            {
                "position": list(p.position),
                "rotation_euler": list(p.rotation_euler),
            }
            for p in self.poses
        ]


class TrajectoryOptimizer:
    """Implements the frontier-based trajectory optimization loop."""

    def __init__(
        self,
        blend_path: Path,
        exclude_keywords: Iterable[str] = ("PointLamp", "Window", "Door", "CeilingLight"),
        target_limit: Optional[int] = None,
        camera_height: float = 1.5,
        min_view_distance: float = 0.5,
        max_view_distance: float = 3.0,
        samples_per_target: int = 2000,
        fov_margin: float = 0.96,
        nav_grid_resolution: float = 0.8,
    ):
        self.blend_path = Path(blend_path)
        self.exclude_keywords = tuple(exclude_keywords)
        self.camera_height = camera_height
        self.min_view_distance = min_view_distance
        self.max_view_distance = max_view_distance
        self.samples_per_target = samples_per_target
        self.fov_margin = fov_margin
        self.nav_grid_resolution = nav_grid_resolution
        self.target_limit = target_limit

        self._load_scene()
        self.scene = bpy.context.scene
        self.depsgraph = bpy.context.evaluated_depsgraph_get()
        self.base_camera = self._ensure_temp_camera()
        self.targets = self._collect_target_objects()
        self.room_polygon = self._compute_room_polygon()
        self.obstacle_polygons = [self._object_xy_polygon(obj) for obj in self.targets]
        self.obstacle_union = unary_union(self.obstacle_polygons) if self.obstacle_polygons else None
        self.nav_graph = self._build_navigation_graph()

    def _load_scene(self):
        if not self.blend_path.exists():
            raise FileNotFoundError(self.blend_path)
        bpy.ops.wm.open_mainfile(filepath=str(self.blend_path))

    def _ensure_temp_camera(self) -> bpy.types.Object:
        scene = bpy.context.scene
        base_camera = scene.camera
        if base_camera is None:
            cam_data = bpy.data.cameras.new("TrajectoryCam")
            base_camera = bpy.data.objects.new("TrajectoryCam", cam_data)
            scene.collection.objects.link(base_camera)
            scene.camera = base_camera

        temp_data = base_camera.data.copy()
        temp_camera = bpy.data.objects.new("TrajectoryOptimizerCamera", temp_data)
        scene.collection.objects.link(temp_camera)
        temp_camera.hide_viewport = True
        temp_camera.hide_render = True
        return temp_camera

    def _collect_target_objects(self) -> list[bpy.types.Object]:
        targets = []
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            if any(keyword in obj.name for keyword in self.exclude_keywords):
                continue
            if obj.parent and obj.parent.type != "MESH":
                continue
            targets.append(obj)

        targets.sort(key=lambda o: abs(self._object_xy_polygon(o).area), reverse=True)
        if self.target_limit is not None:
            targets = targets[: self.target_limit]
        logger.info("Found %d candidate target objects", len(targets))
        return targets

    def _object_xy_polygon(self, obj: bpy.types.Object) -> Polygon:
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        pts = [(c.x, c.y) for c in corners]
        return Polygon(pts).convex_hull

    def _compute_room_polygon(self) -> Polygon:
        all_points = []
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            all_points.extend((c.x, c.y) for c in corners)

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        padding = 0.25
        polygon = box(min_x - padding, min_y - padding, max_x + padding, max_y + padding)
        return polygon

    def _build_navigation_graph(self) -> nx.Graph:
        free_polygon = self.room_polygon
        min_x, min_y, max_x, max_y = free_polygon.bounds
        xs = np.arange(min_x, max_x + self.nav_grid_resolution, self.nav_grid_resolution)
        ys = np.arange(min_y, max_y + self.nav_grid_resolution, self.nav_grid_resolution)

        graph = nx.Graph()
        for x in xs:
            for y in ys:
                p = Point(x, y)
                if not free_polygon.contains(p):
                    continue
                if self.obstacle_union is not None and self.obstacle_union.buffer(0.05).contains(p):
                    continue
                graph.add_node((x, y))

        neighbors = [
            (0, self.nav_grid_resolution),
            (self.nav_grid_resolution, 0),
            (0, -self.nav_grid_resolution),
            (-self.nav_grid_resolution, 0),
            (self.nav_grid_resolution, self.nav_grid_resolution),
            (-self.nav_grid_resolution, self.nav_grid_resolution),
            (self.nav_grid_resolution, -self.nav_grid_resolution),
            (-self.nav_grid_resolution, -self.nav_grid_resolution),
        ]

        for node in list(graph.nodes):
            for dx, dy in neighbors:
                neighbor = (node[0] + dx, node[1] + dy)
                if neighbor not in graph:
                    continue
                segment = LineString([node, neighbor])
                if self.obstacle_union is not None and self.obstacle_union.buffer(0.05).intersects(segment):
                    continue
                weight = segment.length
                graph.add_edge(node, neighbor, weight=weight)

        logger.info("Navigation graph has %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    def optimize(self) -> TrajectoryResult:
        if not self.targets:
            raise RuntimeError("No target objects found for trajectory optimization.")

        current_location = np.array(bpy.context.scene.camera.location[:3])
        visited: set[str] = set()
        poses: list[CameraPose] = []

        for iteration in range(len(self.targets)):
            target = self._select_next_target(current_location, visited)
            if target is None:
                break

            viewpoint = self._find_best_viewpoint(current_location, target)
            if viewpoint is None:
                logger.warning("Could not find feasible viewpoint for %s", target.name)
                visited.add(target.name)
                continue

            path_points = self._plan_path(current_location[:2], viewpoint["position"][:2])
            poses.extend(self._path_to_camera_poses(path_points))
            poses.append(
                CameraPose(
                    position=tuple(viewpoint["position"]),
                    rotation_euler=tuple(viewpoint["rotation"]),
                )
            )

            current_location = viewpoint["position"]
            visited.add(target.name)
            logger.info("Visited %s (%d/%d)", target.name, len(visited), len(self.targets))

        return TrajectoryResult(poses=poses)

    def _select_next_target(self, current_location: np.ndarray, visited: set[str]) -> Optional[bpy.types.Object]:
        remaining = [obj for obj in self.targets if obj.name not in visited]
        if not remaining:
            return None

        distances = []
        for obj in remaining:
            centroid = np.mean([obj.matrix_world @ Vector(corner) for corner in obj.bound_box], axis=0)
            d = np.linalg.norm(current_location[:2] - np.array([centroid[0], centroid[1]]))
            distances.append((d, obj))
        distances.sort(key=lambda item: item[0])
        return distances[0][1]

    def _find_best_viewpoint(self, current_location: np.ndarray, target: bpy.types.Object) -> Optional[dict]:
        best = None
        target_center = np.mean([target.matrix_world @ Vector(corner) for corner in target.bound_box], axis=0)
        target_center = np.array([target_center[0], target_center[1], target_center[2]])

        for sample_idx in range(self.samples_per_target):
            angle = random.uniform(0.0, 2 * math.pi)
            distance = random.uniform(self.min_view_distance, self.max_view_distance)
            candidate_xy = np.array(
                [target_center[0] + distance * math.cos(angle), target_center[1] + distance * math.sin(angle)]
            )
            candidate = np.array([candidate_xy[0], candidate_xy[1], self.camera_height])

            if not self._is_accessible(candidate_xy):
                continue

            rotation = self._look_at_rotation(candidate, target_center)
            coverage = self._coverage_score(candidate, rotation, target)
            if coverage < self.fov_margin:
                continue

            occluded = self._is_occluded(candidate, target)
            if occluded:
                continue

            score = coverage - 0.1 * np.linalg.norm(candidate - current_location)
            if best is None or score > best["score"]:
                best = {
                    "position": candidate,
                    "rotation": rotation,
                    "score": score,
                }

        return best

    def _is_accessible(self, xy_point: np.ndarray) -> bool:
        point = Point(float(xy_point[0]), float(xy_point[1]))
        if not self.room_polygon.contains(point):
            return False
        if self.obstacle_union is not None and self.obstacle_union.buffer(0.05).contains(point):
            return False
        return True

    def _look_at_rotation(self, camera_position: np.ndarray, target_position: np.ndarray) -> tuple[float, float, float]:
        cam_vec = Vector(camera_position)
        tgt_vec = Vector(target_position)
        direction = (tgt_vec - cam_vec).normalized()
        quat = direction.to_track_quat("-Z", "Y")
        euler = quat.to_euler("XYZ")
        return (euler.x, euler.y, euler.z)

    def _coverage_score(self, camera_position: np.ndarray, rotation_euler: tuple[float, float, float], target: bpy.types.Object) -> float:
        cam = self.base_camera
        cam.location = Vector(camera_position)
        cam.rotation_euler = rotation_euler

        bounding_corners = [target.matrix_world @ Vector(corner) for corner in target.bound_box]
        scene = bpy.context.scene
        in_view = 0
        for corner in bounding_corners:
            co_ndc = world_to_camera_view(scene, cam, corner)
            if (
                0.0 <= co_ndc.x <= 1.0
                and 0.0 <= co_ndc.y <= 1.0
                and co_ndc.z >= 0.0
            ):
                in_view += 1
        return in_view / len(bounding_corners)

    def _is_occluded(self, camera_position: np.ndarray, target: bpy.types.Object) -> bool:
        target_center = np.mean([target.matrix_world @ Vector(corner) for corner in target.bound_box], axis=0)
        origin = Vector(camera_position)
        direction = Vector((target_center[0] - origin.x, target_center[1] - origin.y, target_center[2] - origin.z))
        distance = direction.length
        if distance < 1e-4:
            return False
        direction.normalize()

        success, _location, _normal, _face_index, hit_obj, _matrix = bpy.context.scene.ray_cast(
            self.depsgraph,
            origin,
            direction,
            distance=distance - 0.05,
        )
        return success and hit_obj != target

    def _plan_path(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> list[np.ndarray]:
        graph = self.nav_graph.copy()
        start = tuple(start_xy.tolist())
        goal = tuple(goal_xy.tolist())

        for node in (start, goal):
            if node not in graph:
                graph.add_node(node)
                self._connect_node_to_graph(graph, node)

        try:
            path = nx.dijkstra_path(graph, start, goal, weight="weight")
        except nx.NetworkXNoPath as exc:
            raise RuntimeError("Failed to find path between camera poses") from exc

        return [np.array([x, y, self.camera_height]) for x, y in path]

    def _connect_node_to_graph(self, graph: nx.Graph, node: tuple[float, float], k: int = 8):
        nodes = list(graph.nodes)
        nodes.remove(node)
        nodes.sort(key=lambda n: math.dist(n, node))
        connected = 0
        for neighbor in nodes:
            segment = LineString([node, neighbor])
            if self.obstacle_union is not None and self.obstacle_union.buffer(0.05).intersects(segment):
                continue
            graph.add_edge(node, neighbor, weight=segment.length)
            connected += 1
            if connected >= k:
                break

    def _path_to_camera_poses(self, path_points: list[np.ndarray]) -> list[CameraPose]:
        poses: list[CameraPose] = []
        for idx, point in enumerate(path_points[:-1]):
            next_point = path_points[idx + 1]
            direction = next_point - point
            yaw = math.atan2(direction[1], direction[0])
            rotation = (0.0, 0.0, yaw)
            poses.append(CameraPose(position=tuple(point), rotation_euler=rotation))
        return poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize camera trajectory for a Blender scene.")
    parser.add_argument("--blend", type=Path, required=True, help="Path to the input .blend file.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the resulting trajectory JSON.")
    parser.add_argument("--targets", type=int, default=None, help="Optional limit on number of targets to visit.")
    parser.add_argument("--grid", type=float, default=0.8, help="Grid resolution (meters) for Dijkstra planning.")
    parser.add_argument("--samples", type=int, default=2000, help="Candidate viewpoints sampled per target.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    optimizer = TrajectoryOptimizer(
        blend_path=args.blend,
        target_limit=args.targets,
        samples_per_target=args.samples,
        nav_grid_resolution=args.grid,
    )
    trajectory = optimizer.optimize()
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(trajectory.to_json(), f, indent=2)
    logger.info("Trajectory with %d poses written to %s", len(trajectory.poses), args.output)


if __name__ == "__main__":
    main()
