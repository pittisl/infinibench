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
import csv
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import bmesh
import bpy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyrender
import trimesh
from PIL import Image
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector
from scipy.spatial.transform import Rotation as R
from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon, box, MultiPoint
from shapely.ops import unary_union
from shapely.prepared import PreparedGeometry, prep

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


@dataclass
class BatchTrajectoryConfig:
    room_type: str = "living-room_0/0.ceiling"
    frame_prefix: str = "frame_"
    video_filename: str = "trajectory_video.mp4"
    grid_resolution: float = 0.2
    rotation_steps: int = 30
    camera_height: float = 1.5
    exclusion_keywords: Sequence[str] = ("PointLamp", "Window", "Door", "CeilingLight", "Carnivore")
    samples_per_object: int = 500
    min_view_distance: float = 0.2
    max_view_distance: float = 2.0
    max_sight_length: float = 4.0
    occlusion_threshold: float = 0.7
    max_occlusion_checks: int = 40
    robot_radius: float = 0.1
    render_resolution_x: int = 480
    frame_step: int = 3


def triangulate_object(obj: bpy.types.Object) -> None:
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()


def blender_mesh_to_trimesh(obj: bpy.types.Object) -> trimesh.Trimesh:
    triangulate_object(obj)
    mesh = obj.data
    vertices_world = np.array([(obj.matrix_world @ v.co)[:] for v in mesh.vertices], dtype=float)
    faces = np.array([tuple(poly.vertices) for poly in mesh.polygons if len(poly.vertices) == 3], dtype=int)
    return trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)


def mesh_to_xy_polygon(mesh: trimesh.Trimesh) -> Polygon:
    projected = mesh.vertices[:, :2]
    triangles = []
    for face in mesh.faces:
        pts = projected[face]
        if np.abs(np.cross(pts[1] - pts[0], pts[2] - pts[0])) < 1e-6:
            continue
        triangles.append(Polygon(pts))
    if not triangles:
        return Polygon()
    return unary_union(triangles)


def get_unique_color(index: int) -> list[int]:
    r = index & 255
    g = (index >> 8) & 255
    b = (index >> 16) & 255
    return [r, g, b, 255]


def is_valid_point(
    x: float,
    y: float,
    room: PreparedGeometry,
    obstacles: Optional[PreparedGeometry],
) -> bool:
    point = Point(x, y)
    if not room.contains(point):
        return False
    if obstacles and obstacles.intersects(point):
        return False
    return True


def get_nearest_grid_node(x: float, y: float, graph: nx.Graph) -> tuple[float, float]:
    if graph.number_of_nodes() == 0:
        raise ValueError("Navigation graph is empty.")
    nodes = np.array(list(graph.nodes), dtype=float)
    dist = np.sum((nodes - np.array([x, y])) ** 2, axis=1)
    idx = int(np.argmin(dist))
    return tuple(nodes[idx])


def compute_path_segment(start_x: float, start_y: float, end_x: float, end_y: float, graph: nx.Graph) -> list[tuple[float, float]]:
    start = get_nearest_grid_node(start_x, start_y, graph)
    end = get_nearest_grid_node(end_x, end_y, graph)
    try:
        return nx.shortest_path(graph, source=start, target=end, weight="weight")
    except nx.NetworkXNoPath:
        logger.warning("No path found between (%s, %s) and (%s, %s)", start_x, start_y, end_x, end_y)
        return []


def get_contour(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    coords = [(v.co.x, v.co.y) for v in obj.data.vertices]
    if len(coords) < 3:
        return np.array([]), np.array([])
    hull = MultiPoint(coords).convex_hull
    if hull.is_empty or hull.geom_type != "Polygon":
        arr = np.array(coords)
        return arr[:, 0], arr[:, 1]
    xs, ys = hull.exterior.coords.xy
    return np.array(xs), np.array(ys)


def contour2poly(x_vals: np.ndarray, y_vals: np.ndarray) -> Polygon:
    coords = list(zip(x_vals, y_vals))
    if len(coords) < 3:
        raise ValueError("Cannot create polygon with fewer than three points")
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.convex_hull
    return poly


def camera_traj(
    blend_path: Path,
    output_dir: Path,
    config: BatchTrajectoryConfig,
) -> dict:
    """Batch-oriented trajectory optimization translated from the research notebook."""

    blend_path = Path(blend_path)
    output_dir = Path(output_dir)
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_keywords = tuple(config.exclusion_keywords)

    room_obj = next((obj for obj in bpy.data.objects if config.room_type in obj.name), None)
    if room_obj is None:
        raise RuntimeError(f"Could not find object matching room type '{config.room_type}'.")

    x_vals, y_vals = get_contour(room_obj)
    if x_vals.size == 0:
        raise RuntimeError(f"Failed to extract contour for {room_obj.name}")
    poly_room = contour2poly(x_vals, y_vals)
    poly_room = translate(poly_room, xoff=room_obj.location.x, yoff=room_obj.location.y)
    accessible_room = poly_room.buffer(-0.1)
    if accessible_room.is_empty:
        accessible_room = poly_room

    min_dist = float("inf")
    door_location: Optional[tuple[float, float]] = None
    for obj in bpy.data.objects:
        if "doorfactory" not in obj.name.lower():
            continue
        loc = obj.matrix_world.translation
        dist = poly_room.boundary.distance(Point(loc.x, loc.y))
        if dist < min_dist:
            min_dist = dist
            door_location = (loc.x, loc.y)
    if door_location is None:
        raise RuntimeError("Unable to locate a doorfactory object inside the scene.")
    x_start, y_start = door_location

    scene = bpy.context.scene
    if scene.camera is None or scene.camera.data is None:
        raise RuntimeError("Scene camera is required to compute FOV.")
    camera_data = scene.camera.data
    if camera_data.type != "PERSP":
        raise RuntimeError("Perspective camera required for notebook trajectory pipeline.")
    sensor_width = camera_data.sensor_width
    sensor_height = camera_data.sensor_height
    focal_length = camera_data.lens
    hor_fov_rad = 2 * math.atan((sensor_width / 2) / focal_length)
    ver_fov_rad = 2 * math.atan((sensor_height / 2) / focal_length)

    bpy.ops.object.mode_set(mode="OBJECT")
    target_poly_density = 100000
    obj_dimensions_data = []
    spawn_meshes: list[trimesh.Trimesh] = []
    spawn_polygons: list[Polygon] = []
    spawn_names: list[str] = []

    for obj in bpy.data.objects:
        if obj.type != "MESH" or "spawn_asset" not in obj.name:
            continue
        if any(keyword in obj.name for keyword in exclude_keywords):
            continue

        polygon_area = sum(poly.area for poly in obj.data.polygons) or 1.0
        cur_density = len(obj.data.polygons) / polygon_area
        if cur_density > target_poly_density:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            mod = obj.modifiers.new(name="TrajectoryDecimate", type="DECIMATE")
            mod.ratio = target_poly_density / cur_density
            mod.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=mod.name)
            bpy.ops.object.select_all(action="DESELECT")

        try:
            tm = blender_mesh_to_trimesh(obj)
            poly_2d = mesh_to_xy_polygon(tm)
        except Exception as exc:  # pragma: no cover - Blender context only
            logger.warning("Skipping %s due to mesh conversion error: %s", obj.name, exc)
            continue

        if poly_2d.is_empty:
            continue

        spawn_meshes.append(tm)
        spawn_polygons.append(poly_2d)
        spawn_names.append(obj.name)
        obj_dimensions_data.append(
            {
                "name": obj.name,
                "width_x": tm.extents[0],
                "length_y": tm.extents[1],
                "height_z": tm.extents[2],
            }
        )

    bbox_csv = output_dir / "object_bbox_dimensions.csv"
    with bbox_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "width_x", "length_y", "height_z"])
        writer.writeheader()
        writer.writerows(obj_dimensions_data)

    wall_mesh = None
    for obj in bpy.data.objects:
        if obj.type == "MESH" and "dining-room_0/0.wall" in obj.name:
            if any(keyword in obj.name for keyword in exclude_keywords):
                continue
            wall_mesh = blender_mesh_to_trimesh(obj)
            break

    trimesh_obj_list = spawn_meshes.copy()
    if wall_mesh is not None:
        trimesh_obj_list.append(wall_mesh)

    polygons_2d = [poly for poly in spawn_polygons if not poly.is_empty]

    if not spawn_names or not polygons_2d:
        raise RuntimeError("No valid spawn assets found to build trajectory.")

    captured_obj_idx: list[int] = []
    l_cam_x = [x_start]
    l_cam_y = [y_start]
    l_yaw = [math.pi / 2]
    l_pitch = [0.0]
    cur_cam_x = x_start
    cur_cam_y = y_start
    max_inside = 0.0
    appearance_obj_list: list[dict] = []

    y_resolution = max(1, math.floor(config.render_resolution_x / (math.tan(hor_fov_rad / 2) / math.tan(ver_fov_rad / 2))))

    for obj_cnt in range(len(polygons_2d)):
        distances = []
        for idx, poly in enumerate(polygons_2d):
            if idx in captured_obj_idx:
                distances.append(float("inf"))
                continue
            center = poly.centroid
            distances.append(math.hypot(cur_cam_x - center.x, cur_cam_y - center.y))

        if not distances or min(distances) == float("inf"):
            break

        index_closest = int(np.argmin(distances))
        captured_obj_idx.append(index_closest)

        scene_all = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        scene_target = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])

        for idx, mesh in enumerate(trimesh_obj_list):
            mesh_copy = mesh.copy()
            color = get_unique_color(random.randint(0, (1 << 24) - 1))
            mesh_copy.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
            pm_all = pyrender.Mesh.from_trimesh(mesh_copy, smooth=False)
            scene_all.add(pm_all)
            if idx == index_closest:
                pm_target = pyrender.Mesh.from_trimesh(mesh_copy.copy(), smooth=False)
                scene_target.add(pm_target)

        default_pose = np.eye(4)
        camera_all = pyrender.PerspectiveCamera(yfov=ver_fov_rad)
        camera_target = pyrender.PerspectiveCamera(yfov=ver_fov_rad)
        camera_node = scene_all.add(camera_all, pose=default_pose)
        target_camera_node = scene_target.add(camera_target, pose=default_pose)

        obj_poly = polygons_2d[index_closest]
        sampling_cnt = 0

        while sampling_cnt < config.samples_per_object:
            sampling_cnt += 1
            dis2camera = random.uniform(config.min_view_distance, config.max_view_distance)
            angle_random = random.uniform(0.0, math.pi)
            yaw_random = random.uniform(math.pi / 4, math.pi * 3 / 4)
            pitch_random = random.uniform(0.0, math.pi)

            centroid = obj_poly.centroid
            cam_x = centroid.x + dis2camera * math.cos(angle_random)
            cam_y = centroid.y + dis2camera * math.sin(angle_random)
            candidate_2d = Point(cam_x, cam_y)

            if not accessible_room.contains(candidate_2d):
                continue
            if any(poly.contains(candidate_2d) for poly in polygons_2d):
                continue

            delta_z = config.max_sight_length * math.sin(ver_fov_rad / 2)
            delta_x = config.max_sight_length * math.cos(ver_fov_rad / 2) * math.sin(hor_fov_rad / 2)
            delta_y = config.max_sight_length * math.cos(ver_fov_rad / 2) * math.cos(hor_fov_rad / 2)

            vertices = np.array(
                [
                    [cam_x, cam_y, config.camera_height],
                    [cam_x + delta_x, cam_y + delta_y, config.camera_height + delta_z],
                    [cam_x - delta_x, cam_y + delta_y, config.camera_height + delta_z],
                    [cam_x - delta_x, cam_y + delta_y, config.camera_height - delta_z],
                    [cam_x + delta_x, cam_y + delta_y, config.camera_height - delta_z],
                ]
            )
            faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]])
            fov_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            rotation_pitch = trimesh.transformations.rotation_matrix(
                pitch_random,
                [0, 0, 1],
                point=[cam_x, cam_y, config.camera_height],
            )
            rotation_yaw = trimesh.transformations.rotation_matrix(
                yaw_random - math.pi / 2,
                [1, 0, 0],
                point=[cam_x, cam_y, config.camera_height],
            )
            fov_mesh.apply_transform(rotation_yaw)
            fov_mesh.apply_transform(rotation_pitch)

            points = spawn_meshes[index_closest].vertices
            inside = fov_mesh.contains(points)
            percent = float(np.mean(inside)) * 100 if len(inside) else 0.0
            max_inside = max(max_inside, percent)
            if not inside.all():
                continue

            r = R.from_euler("xyz", [yaw_random, 0.0, pitch_random])
            pose = np.eye(4)
            pose[:3, :3] = r.as_matrix()
            pose[:3, 3] = [cam_x, cam_y, config.camera_height]

            scene_all.set_pose(camera_node, pose=pose)
            scene_target.set_pose(target_camera_node, pose=pose)

            renderer = pyrender.OffscreenRenderer(
                viewport_width=config.render_resolution_x,
                viewport_height=y_resolution,
            )
            try:
                color_all, _ = renderer.render(scene_all)
                color_target, _ = renderer.render(scene_target)
            finally:
                renderer.delete()

            pixels = color_target.reshape(-1, color_target.shape[-1])
            unique_colors = np.unique(pixels, axis=0)
            if unique_colors.size == 0:
                continue
            target_color = unique_colors[-1]

            mask_rendered = np.all(color_all == target_color, axis=-1)
            mask_target = np.all(color_target == target_color, axis=-1)
            cnt_pixels_rendered = int(np.sum(mask_rendered))
            cnt_pixels_all = int(np.sum(mask_target))
            if cnt_pixels_all == 0:
                continue
            percent_showed = cnt_pixels_rendered / cnt_pixels_all

            if percent_showed >= config.occlusion_threshold or sampling_cnt >= config.max_occlusion_checks:
                cur_cam_x = cam_x
                cur_cam_y = cam_y
                l_cam_x.append(cam_x)
                l_cam_y.append(cam_y)
                l_yaw.append(yaw_random)
                l_pitch.append(pitch_random)

                Image.fromarray(color_all).save(output_dir / f"segmentation_obj{obj_cnt}_all.png")
                Image.fromarray(color_target).save(output_dir / f"segmentation_obj{obj_cnt}_target.png")

                appearance_obj_list.append(
                    {
                        "name": spawn_names[index_closest],
                        "occlusion": percent_showed,
                    }
                )

                break

    appearance_csv = output_dir / "object_appearance.csv"
    with appearance_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "occlusion"])
        writer.writeheader()
        writer.writerows(appearance_obj_list)

    robot_radius = config.robot_radius
    obstacles_union = unary_union(polygons_2d)
    obstacles_buffered = obstacles_union.buffer(robot_radius) if not obstacles_union.is_empty else Polygon()
    room_eroded = poly_room.buffer(-robot_radius)
    if room_eroded.is_empty:
        room_eroded = poly_room

    prep_obstacles = prep(obstacles_buffered) if not obstacles_buffered.is_empty else None
    prep_room = prep(room_eroded)

    graph = nx.Graph()
    minx, miny, maxx, maxy = poly_room.bounds
    x_coords = np.arange(minx, maxx, config.grid_resolution)
    y_coords = np.arange(miny, maxy, config.grid_resolution)
    valid_nodes = []
    for x in x_coords:
        for y in y_coords:
            if is_valid_point(x, y, prep_room, prep_obstacles):
                node = (round(x, 3), round(y, 3))
                graph.add_node(node)
                valid_nodes.append(node)

    directions = [
        (config.grid_resolution, 0, 1.0),
        (-config.grid_resolution, 0, 1.0),
        (0, config.grid_resolution, 1.0),
        (0, -config.grid_resolution, 1.0),
        (config.grid_resolution, config.grid_resolution, math.sqrt(2)),
        (config.grid_resolution, -config.grid_resolution, math.sqrt(2)),
        (-config.grid_resolution, config.grid_resolution, math.sqrt(2)),
        (-config.grid_resolution, -config.grid_resolution, math.sqrt(2)),
    ]
    for node in valid_nodes:
        for dx, dy, weight in directions:
            neighbor = (round(node[0] + dx, 3), round(node[1] + dy, 3))
            if neighbor in graph:
                graph.add_edge(node, neighbor, weight=weight)

    full_trajectory: list[dict] = []
    for idx in range(len(l_cam_x) - 1):
        curr_x, curr_y = l_cam_x[idx], l_cam_y[idx]
        next_x, next_y = l_cam_x[idx + 1], l_cam_y[idx + 1]
        curr_yaw, curr_pitch = l_yaw[idx], l_pitch[idx]
        next_yaw, next_pitch = l_yaw[idx + 1], l_pitch[idx + 1]

        path_points = compute_path_segment(curr_x, curr_y, next_x, next_y, graph)
        for point in path_points:
            full_trajectory.append(
                {
                    "x": point[0],
                    "y": point[1],
                    "z": config.camera_height,
                    "yaw": curr_yaw,
                    "pitch": curr_pitch,
                    "action": "translate",
                }
            )

        for step in range(1, config.rotation_steps + 1):
            alpha = step / config.rotation_steps
            interp_yaw = curr_yaw + (next_yaw - curr_yaw) * alpha
            interp_pitch = curr_pitch + (next_pitch - curr_pitch) * alpha
            full_trajectory.append(
                {
                    "x": next_x,
                    "y": next_y,
                    "z": config.camera_height,
                    "yaw": interp_yaw,
                    "pitch": interp_pitch,
                    "action": "rotate",
                }
            )

    plt.figure(figsize=(10, 10))
    rx, ry = poly_room.exterior.xy
    plt.plot(rx, ry, "k-", linewidth=2, label="Room")
    for poly in polygons_2d:
        if poly.is_empty:
            continue
        if poly.geom_type == "Polygon":
            ox, oy = poly.exterior.xy
            plt.fill(ox, oy, color="gray", alpha=0.5)
        elif poly.geom_type == "MultiPolygon":
            for sub in poly.geoms:
                ox, oy = sub.exterior.xy
                plt.fill(ox, oy, color="gray", alpha=0.5)
    if full_trajectory:
        traj_x = [state["x"] for state in full_trajectory]
        traj_y = [state["y"] for state in full_trajectory]
        plt.plot(traj_x, traj_y, "b.-", markersize=2, label="Computed Path")
    plt.plot(l_cam_x, l_cam_y, "ro", markersize=6, label="Keypoints")
    plt.title("Dijkstra Path Planning (Translate -> Rotate)")
    plt.axis("equal")
    plt.legend()
    plt.savefig(output_dir / "visual.pdf")
    plt.close()

    cam = bpy.data.objects.get("camera_0_0") or scene.camera
    if cam is None:
        raise RuntimeError("Unable to locate camera object for animation.")
    if cam.rotation_mode != "XYZ":
        cam.rotation_mode = "XYZ"

    if full_trajectory:
        scene.frame_start = 0
        scene.frame_end = len(full_trajectory) - 1

    trajectory_csv = output_dir / "trajectory_data.csv"
    if full_trajectory:
        with trajectory_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=full_trajectory[0].keys())
            writer.writeheader()
            writer.writerows(full_trajectory)
    else:
        logger.warning("Full trajectory is empty for %s", blend_path)

    for frame_idx, state in enumerate(full_trajectory):
        scene.frame_set(frame_idx)
        x, y, z = state["x"], state["y"], state["z"]
        yaw, pitch = state["yaw"], state["pitch"]
        rot_euler = Euler((yaw, 0.0, pitch), "XYZ")
        mat_rot = rot_euler.to_matrix().to_4x4()
        mat_loc = Matrix.Translation((x, y, z))
        cam.matrix_world = mat_loc @ mat_rot
        cam.keyframe_insert(data_path="location", index=-1)
        cam.keyframe_insert(data_path="rotation_euler", index=-1)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        scene.cycles.device = "GPU"
    except Exception:  # pragma: no cover
        scene.cycles.device = "CPU"

    scene.cycles.samples = 64
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_denoising = True
    scene.view_settings.exposure = 1.0
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.frame_step = config.frame_step
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_dir / config.frame_prefix)
    bpy.ops.render.render(animation=True)

    return {
        "blend_path": str(blend_path),
        "output_dir": str(output_dir),
        "trajectory_frames": len(full_trajectory),
        "bbox_csv": str(bbox_csv),
        "appearance_csv": str(appearance_csv),
        "trajectory_csv": str(trajectory_csv) if full_trajectory else None,
    }


def run_batch_pipeline(
    folder_path: Path,
    output_root: Optional[Path],
    config: BatchTrajectoryConfig,
) -> list[dict]:
    folder_path = Path(folder_path)
    if output_root is None:
        output_root = folder_path
    else:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    blend_files = sorted(folder_path.glob("*.blend"))
    if not blend_files:
        raise RuntimeError(f"No .blend files found under {folder_path}")

    summaries = []
    for blend_file in blend_files:
        out_dir = output_root / blend_file.stem
        logger.info("Processing %s -> %s", blend_file, out_dir)
        summary = camera_traj(blend_file, out_dir, config)
        summaries.append(summary)
    return summaries


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
    parser.add_argument("--blend", type=Path, help="Path to the input .blend file.")
    parser.add_argument("--output", type=Path, help="Path to write the resulting trajectory JSON.")
    parser.add_argument("--targets", type=int, default=None, help="Optional limit on number of targets to visit.")
    parser.add_argument("--grid", type=float, default=0.8, help="Grid resolution (meters) for Dijkstra planning.")
    parser.add_argument("--samples", type=int, default=2000, help="Candidate viewpoints sampled per target.")
    parser.add_argument("--batch-folder", type=Path, help="Process every .blend file in this folder via the notebook pipeline.")
    parser.add_argument("--batch-output", type=Path, help="Optional output root for batch processing (defaults to folder).")
    parser.add_argument("--batch-room-type", default="living-room_0/0.ceiling", help="Room object identifier for the batch pipeline.")
    parser.add_argument("--batch-frame-prefix", default="frame_", help="Filename prefix for rendered frames (batch mode).")
    parser.add_argument("--batch-video-name", default="trajectory_video.mp4", help="Video filename placeholder (batch mode).")
    parser.add_argument("--batch-grid", type=float, default=0.2, help="Grid resolution for the batch navigation graph.")
    parser.add_argument("--batch-rotation-steps", type=int, default=30, help="Rotation interpolation steps for batch outputs.")
    parser.add_argument("--batch-samples", type=int, default=500, help="Samples per object for the batch viewpoint search.")
    parser.add_argument("--batch-height", type=float, default=1.5, help="Camera height for the batch pipeline.")
    parser.add_argument("--batch-min-distance", type=float, default=0.2, help="Minimum camera distance during batch sampling.")
    parser.add_argument("--batch-max-distance", type=float, default=2.0, help="Maximum camera distance during batch sampling.")
    parser.add_argument("--batch-occlusion", type=float, default=0.7, help="Visibility threshold for accepting a batch sample.")
    parser.add_argument("--batch-occlusion-checks", type=int, default=40, help="Maximum occlusion checks per object for batch mode.")
    parser.add_argument("--batch-frame-step", type=int, default=3, help="Frame step used when rendering batch animations.")
    parser.add_argument("--batch-resolution", type=int, default=480, help="Horizontal resolution for batch segmentation renders.")
    parser.add_argument("--batch-max-sight", type=float, default=4.0, help="Max sight length for FOV sampling in batch mode.")
    parser.add_argument("--batch-robot-radius", type=float, default=0.1, help="Robot radius buffer for batch navigation.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.batch_folder:
        config = BatchTrajectoryConfig(
            room_type=args.batch_room_type,
            frame_prefix=args.batch_frame_prefix,
            video_filename=args.batch_video_name,
            grid_resolution=args.batch_grid,
            rotation_steps=args.batch_rotation_steps,
            camera_height=args.batch_height,
            samples_per_object=args.batch_samples,
            min_view_distance=args.batch_min_distance,
            max_view_distance=args.batch_max_distance,
            occlusion_threshold=args.batch_occlusion,
            max_occlusion_checks=args.batch_occlusion_checks,
            frame_step=args.batch_frame_step,
            render_resolution_x=args.batch_resolution,
            max_sight_length=args.batch_max_sight,
            robot_radius=args.batch_robot_radius,
        )
        summaries = run_batch_pipeline(args.batch_folder, args.batch_output, config)
        logger.info("Batch completed for %d scenes", len(summaries))
        return

    if args.blend is None or args.output is None:
        raise SystemExit("--blend and --output are required unless --batch-folder is provided.")

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
