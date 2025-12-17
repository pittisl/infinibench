#!/usr/bin/env python3
"""End-to-end InfiniBench pipeline (text -> scene -> video -> QA -> metrics)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from infinigen_examples.qa_from_metadata import generate_tasks

LOGGER = logging.getLogger("infinibench.pipeline")
REPO_ROOT = Path(__file__).resolve().parents[1]
NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-description", required=True, help="Natural-language scene request passed to the agent.")
    parser.add_argument("--blender", default="blender", help="Path to the Blender executable.")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory to store all stage outputs (default: runs/infinibench_<timestamp>).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed forwarded to generate_indoors.")
    parser.add_argument("--disable-agentic", action="store_true", help="Skip agentic constraint synthesis even when a description is provided.")
    parser.add_argument("--agentic-max-iterations", type=int, default=3, help="Max refinement loops for the agentic constraint generator.")
    parser.add_argument("--gin-config", action="append", default=[], help="Additional gin config files to load (exclude .gin suffix).")
    parser.add_argument("--gin-override", action="append", default=[], help="Extra gin overrides forwarded to generate_indoors.")

    # Trajectory-specific knobs (batch pipeline only)
    parser.add_argument("--room-tag", default="living-room_0/0.ceiling", help="Room identifier passed to the trajectory batch pipeline.")
    parser.add_argument("--frame-prefix", default="trajectory_frame_", help="Prefix for rendered trajectory frames.")
    parser.add_argument("--trajectory-samples", type=int, default=500, help="Samples per object during viewpoint search.")
    parser.add_argument("--trajectory-grid", type=float, default=0.2, help="Grid resolution (meters) used for navigation.")
    parser.add_argument("--trajectory-height", type=float, default=1.5, help="Camera height in meters.")
    parser.add_argument("--trajectory-min-distance", type=float, default=0.2, help="Minimum camera distance from each object.")
    parser.add_argument("--trajectory-max-distance", type=float, default=2.0, help="Maximum camera distance from each object.")
    parser.add_argument("--trajectory-occlusion", type=float, default=0.7, help="Visibility threshold for accepting a candidate viewpoint.")
    parser.add_argument("--trajectory-frame-step", type=int, default=3, help="Frame step when rendering trajectory animations.")
    parser.add_argument("--trajectory-resolution", type=int, default=640, help="Horizontal resolution for rendered frames.")
    parser.add_argument("--trajectory-max-sight", type=float, default=4.0, help="Maximum sight length for field-of-view sampling.")
    parser.add_argument("--trajectory-robot-radius", type=float, default=0.1, help="Robot radius buffer for navigation graph inflation.")

    # QA and evaluation stages
    parser.add_argument("--measurement-tasks", type=int, default=5, help="Number of measurement QA tasks to emit.")
    parser.add_argument("--perspective-tasks", type=int, default=5, help="Number of perspective QA tasks to emit.")
    parser.add_argument("--spatiotemporal-tasks", type=int, default=3, help="Number of spatiotemporal QA tasks to emit.")
    parser.add_argument("--qa-seed", type=int, default=1234, help="Seed controlling QA task sampling.")
    parser.add_argument("--responses", type=Path, default=None, help="Optional JSON file containing model predictions to score.")

    # Video encoding helpers
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable used to assemble frames into a video.")
    parser.add_argument("--video-fps", type=int, default=24, help="Frame rate for the encoded video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    output_root = resolve_output_root(args.output_root)
    scene_dir = output_root / "scene"
    trajectory_root = output_root / "trajectory"
    qa_dir = output_root / "qa"
    metrics_path = output_root / "metrics.json"

    LOGGER.info("Writing artifacts to %s", output_root)
    blend_path = stage_scene_generation(args, scene_dir)
    metadata_dir = stage_trajectory(args, blend_path, trajectory_root)
    video_path = encode_video(trajectory_root / blend_path.stem, args.frame_prefix, args.ffmpeg_bin, args.video_fps)
    qa_payload, qa_path = stage_qa(args, metadata_dir, qa_dir)
    metrics = stage_metrics(args, qa_payload["tasks"], metrics_path)

    LOGGER.info("Pipeline complete.")
    LOGGER.info("Scene blend: %s", blend_path)
    LOGGER.info("Trajectory metadata: %s", metadata_dir)
    if video_path:
        LOGGER.info("Trajectory video: %s", video_path)
    else:
        LOGGER.info("Trajectory video: skipped (frames available under %s)", metadata_dir)
    LOGGER.info("QA spec: %s", qa_path)
    if metrics:
        LOGGER.info("Metrics: %s", metrics_path)
    else:
        LOGGER.info("Metrics: skipped (no responses provided)")


def resolve_output_root(user_path: Path | None) -> Path:
    if user_path is not None:
        root = user_path.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path("runs") / f"infinibench_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def shlex_join(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_subprocess(cmd: list[str]) -> None:
    LOGGER.info("Executing: %s", shlex_join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def stage_scene_generation(args: argparse.Namespace, scene_dir: Path) -> Path:
    scene_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.blender,
        "--background",
        "--python",
        str(REPO_ROOT / "infinigen_examples/generate_indoors.py"),
        "--",
        "--output_folder",
        str(scene_dir),
    ]

    if args.seed is not None:
        cmd.extend(["-s", str(args.seed)])

    gin_configs = ["base", *args.gin_config]
    if gin_configs:
        cmd.append("-g")
        cmd.extend(gin_configs)

    overrides = list(args.gin_override)
    description_literal = json.dumps(args.scene_description)
    overrides.append(f"compose_indoors.scene_description={description_literal}")
    overrides.append(
        f"compose_indoors.use_agentic_constraints={'False' if args.disable_agentic else 'True'}"
    )
    overrides.append(
        f"compose_indoors.agentic_max_iterations={int(args.agentic_max_iterations)}"
    )

    if overrides:
        cmd.append("-p")
        cmd.extend(overrides)

    run_subprocess(cmd)

    blend_path = scene_dir / "scene.blend"
    if not blend_path.exists():
        raise FileNotFoundError(f"Expected Blender scene at {blend_path}")
    return blend_path


def stage_trajectory(args: argparse.Namespace, blend_path: Path, trajectory_root: Path) -> Path:
    trajectory_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.blender,
        "--background",
        "--python",
        str(REPO_ROOT / "infinigen_examples/trajectory_optimizer.py"),
        "--",
        "--batch-folder",
        str(blend_path.parent),
        "--batch-output",
        str(trajectory_root),
        "--batch-room-type",
        args.room_tag,
        "--batch-frame-prefix",
        args.frame_prefix,
        "--batch-samples",
        str(args.trajectory_samples),
        "--batch-grid",
        str(args.trajectory_grid),
        "--batch-height",
        str(args.trajectory_height),
        "--batch-min-distance",
        str(args.trajectory_min_distance),
        "--batch-max-distance",
        str(args.trajectory_max_distance),
        "--batch-occlusion",
        str(args.trajectory_occlusion),
        "--batch-frame-step",
        str(args.trajectory_frame_step),
        "--batch-resolution",
        str(args.trajectory_resolution),
        "--batch-max-sight",
        str(args.trajectory_max_sight),
        "--batch-robot-radius",
        str(args.trajectory_robot_radius),
    ]
    run_subprocess(cmd)

    metadata_dir = trajectory_root / blend_path.stem
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Trajectory optimizer did not create {metadata_dir}. Check Blender logs above."
        )
    required = [
        metadata_dir / "object_bbox_dimensions.csv",
        metadata_dir / "object_appearance.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing expected metadata file: {path}")
    return metadata_dir


def encode_video(frames_dir: Path, frame_prefix: str, ffmpeg_bin: str, fps: int) -> Path | None:
    frames = sorted(frames_dir.glob(f"{frame_prefix}*.png"))
    if not frames:
        LOGGER.warning("No rendered frames found under %s", frames_dir)
        return None

    if shutil.which(ffmpeg_bin) is None:
        LOGGER.warning("ffmpeg binary '%s' not found. Skipping video encoding.", ffmpeg_bin)
        return None

    video_path = frames_dir / "trajectory_video.mp4"

    digits = 4
    sample_name = frames[0].name
    suffix = sample_name[len(frame_prefix) : sample_name.rfind(".")]
    if suffix.isdigit():
        digits = len(suffix)

    pattern = str(frames_dir / f"{frame_prefix}%0{digits}d.png")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]

    try:
        LOGGER.info("Encoding video via: %s", shlex_join(cmd))
        subprocess.run(cmd, cwd=frames_dir, check=True)
        return video_path
    except subprocess.CalledProcessError as exc:  # pragma: no cover - integration point
        LOGGER.warning("ffmpeg failed (%s). Frames remain at %s", exc, frames_dir)
        return None


def stage_qa(
    args: argparse.Namespace,
    metadata_dir: Path,
    qa_dir: Path,
) -> tuple[dict[str, Any], Path]:
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_payload = generate_tasks(
        metadata_dir=metadata_dir,
        measurement=args.measurement_tasks,
        perspective=args.perspective_tasks,
        spatiotemporal=args.spatiotemporal_tasks,
        seed=args.qa_seed,
    )
    qa_path = qa_dir / "qa_tasks.json"
    with qa_path.open("w", encoding="utf-8") as f:
        json.dump(qa_payload, f, indent=2)
    return qa_payload, qa_path


def stage_metrics(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    metrics_path: Path,
) -> dict[str, Any] | None:
    if args.responses is None:
        return None

    prediction_map = load_predictions(args.responses)
    metrics = score_predictions(tasks, prediction_map)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def load_predictions(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for container_key in ("responses", "tasks", "predictions"):
            if container_key in data and isinstance(data[container_key], list):
                return records_to_mapping(data[container_key])
        if all(isinstance(v, (int, float, str)) for v in data.values()):
            return {str(k): v for k, v in data.items()}
        return {str(k): extract_prediction_value(v) for k, v in data.items()}

    if isinstance(data, list):
        return records_to_mapping(data)

    raise TypeError("Unsupported prediction file format.")


def records_to_mapping(records: Iterable[Any]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Prediction entries must be JSON objects with an 'id'.")
        task_id = record.get("id") or record.get("task_id")
        if task_id is None:
            raise KeyError("Missing 'id' field in prediction entry.")
        mapping[str(task_id)] = extract_prediction_value(record)
    return mapping


def extract_prediction_value(record: Any) -> Any:
    if isinstance(record, dict):
        for key in ("prediction", "response", "answer", "value"):
            if key in record:
                return record[key]
        return None
    return record


def score_predictions(tasks: list[dict[str, Any]], predictions: dict[str, Any]) -> dict[str, Any]:
    measurement_scores: list[float] = []
    perspective_scores: list[float] = []
    spatiotemporal_scores: list[float] = []
    missing: list[str] = []
    invalid: list[str] = []

    for task in tasks:
        task_id = task.get("id") or "unknown"
        category = task.get("category")
        target = task.get("answer")
        prediction = predictions.get(task_id)

        if prediction is None:
            missing.append(task_id)
            continue

        if category in {"measurement", "perspective"}:
            try:
                pred_value = parse_numeric(prediction)
                target_value = float(target)
                epsilon = 1e-3 if category == "measurement" else 1.0
                score = relative_accuracy(target_value, pred_value, epsilon)
                if category == "measurement":
                    measurement_scores.append(score)
                else:
                    perspective_scores.append(score)
            except (ValueError, TypeError):
                invalid.append(task_id)
        elif category == "spatiotemporal":
            gold = str(target).strip().lower()
            guess = str(prediction).strip().lower()
            spatiotemporal_scores.append(1.0 if gold == guess else 0.0)
        else:
            invalid.append(task_id)

    return {
        "mean_relative_accuracy": {
            "measurement": average(measurement_scores),
            "perspective": average(perspective_scores),
            "overall": average(measurement_scores + perspective_scores),
        },
        "exact_match_accuracy": {
            "spatiotemporal": average(spatiotemporal_scores),
        },
        "scored_tasks": {
            "measurement": len(measurement_scores),
            "perspective": len(perspective_scores),
            "spatiotemporal": len(spatiotemporal_scores),
        },
        "missing_predictions": missing,
        "invalid_predictions": invalid,
    }


def parse_numeric(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric predictions.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        match = NUMERIC_RE.search(cleaned)
        if match:
            return float(match.group(0))
    raise ValueError(f"Cannot parse numeric value from {value!r}")


def relative_accuracy(target: float, prediction: float, epsilon: float) -> float:
    denom = max(abs(target), epsilon)
    error = abs(prediction - target) / denom
    return max(0.0, 1.0 - error)


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    main()
