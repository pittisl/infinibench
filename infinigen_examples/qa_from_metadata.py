"""Generate QA tasks from trajectory optimizer metadata.

The generated tasks follow three categories:
1. Measurement tasks about object dimensions.
2. Perspective-taking tasks (counting from a given trajectory).
3. Spatiotemporal ordering tasks based on object appearance order.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class ObjectMetadata:
    name: str
    width: float
    length: float
    height: float

    @property
    def footprint(self) -> float:
        return self.width * self.length


def load_object_dimensions(csv_path: Path) -> list[ObjectMetadata]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {csv_path}")

    objects: list[ObjectMetadata] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                objects.append(
                    ObjectMetadata(
                        name=row["name"],
                        width=float(row["width_x"]),
                        length=float(row["length_y"]),
                        height=float(row["height_z"]),
                    )
                )
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Invalid row in {csv_path}: {row}") from exc
    if not objects:
        raise ValueError(f"No object entries found in {csv_path}")
    return objects


def load_object_appearance(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []
    names: list[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "name" in row:
                names.append(row["name"])
    return names


def prettify_name(name: str) -> str:
    cleaned = re.sub(r"spawn_asset[_-]*", "", name, flags=re.IGNORECASE)
    tokens = re.split(r"[_\-\s]+", cleaned)
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        return name
    return " ".join(word.capitalize() for word in tokens)


def infer_material(name: str) -> str:
    lowered = name.lower()
    mapping = {
        "wood": "warm wood grain",
        "oak": "oaken grain",
        "metal": "brushed metal",
        "steel": "steel",
        "glass": "frosted glass",
        "fabric": "woven fabric",
        "leather": "matte leather",
        "stone": "polished stone",
        "marble": "marble",
        "plastic": "smooth plastic",
        "ceramic": "ceramic glaze",
    }
    for key, material in mapping.items():
        if key in lowered:
            return material
    return "neutral matte finish"


def pluralize(label: str) -> str:
    if label.endswith("s"):
        return label
    if label.endswith("y") and label[-2:] not in ("ay", "ey"):
        return label[:-1] + "ies"
    return label + "s"


def relation_phrase(current: ObjectMetadata, other: ObjectMetadata) -> str:
    if other.height == 0:
        return f"placed near the {prettify_name(other.name)}"
    ratio = current.height / other.height
    if ratio > 1.1:
        return f"that stands taller than the {prettify_name(other.name)}"
    if ratio < 0.9:
        return f"that sits lower than the {prettify_name(other.name)}"
    return f"that is about the same height as the {prettify_name(other.name)}"


def normalize_label(name: str) -> str:
    pretty = prettify_name(name)
    pretty = re.sub(r"\d+", "", pretty).strip()
    return pretty or name


def generate_measurement_tasks(
    objects: Sequence[ObjectMetadata],
    max_tasks: int,
) -> list[dict]:
    if len(objects) < 2:
        return []
    tasks: list[dict] = []
    sorted_objs = sorted(objects, key=lambda obj: obj.height, reverse=True)
    pairs = zip(sorted_objs, itertools.cycle(sorted_objs[1:] + sorted_objs[:1]))
    for current, other in pairs:
        material = infer_material(current.name)
        relation = relation_phrase(current, other)
        description = prettify_name(current.name)
        question = (
            f"What's the height of the {description} with the texture of {material} "
            f"{relation}?"
        )
        tasks.append(
            {
                "category": "measurement",
                "question": question,
                "answer": round(current.height, 3),
                "units": "meters",
                "ground_truth_field": "height_z",
                "object_name": current.name,
            }
        )
        if len(tasks) >= max_tasks:
            break
    return tasks


def generate_perspective_tasks(
    objects: Sequence[ObjectMetadata],
    max_tasks: int,
) -> list[dict]:
    counts: dict[str, int] = {}
    for obj in objects:
        label = normalize_label(obj.name)
        counts[label] = counts.get(label, 0) + 1

    questions = []
    for label, count in counts.items():
        question = f"How many {label.lower()} are shown in the trajectory video?"
        questions.append(
            {
                "category": "perspective",
                "question": question,
                "answer": count,
                "answer_type": "count",
            }
        )
    random.shuffle(questions)
    return questions[:max_tasks]


def describe_object(name: str) -> str:
    pretty = prettify_name(name)
    material = infer_material(name)
    return f"{pretty} with a {material} finish"


def generate_spatiotemporal_tasks(
    appearance_order: Sequence[str],
    max_tasks: int,
) -> list[dict]:
    if len(appearance_order) < 3:
        return []

    tasks: list[dict] = []
    unique_names = list(dict.fromkeys(appearance_order))
    triplets = []
    for start in range(0, len(unique_names) - 2):
        triplets.append(unique_names[start : start + 3])

    random.shuffle(triplets)
    for triple in triplets:
        descriptions = [describe_object(name) for name in triple]
        question = (
            "What's the appearance order of "
            + ", ".join(descriptions[:-1])
            + f", and {descriptions[-1]}?"
        )

        actual_order = triple
        permutations = list(itertools.permutations(triple))
        random.shuffle(permutations)
        options = []
        seen_orders = set()
        for perm in permutations:
            label = ", ".join(prettify_name(name) for name in perm)
            if label in seen_orders:
                continue
            seen_orders.add(label)
            options.append(label)
            if len(options) == 4:
                break

        answer_label = ", ".join(prettify_name(name) for name in actual_order)
        if answer_label not in options:
            options[0] = answer_label
            random.shuffle(options)

        tasks.append(
            {
                "category": "spatiotemporal",
                "question": question,
                "options": options,
                "answer": answer_label,
                "metric": "exact_match",
            }
        )
        if len(tasks) >= max_tasks:
            break
    return tasks


def generate_tasks(
    metadata_dir: Path,
    measurement: int,
    perspective: int,
    spatiotemporal: int,
    seed: int | None = None,
) -> dict:
    if seed is not None:
        random.seed(seed)

    dimensions_path = metadata_dir / "object_bbox_dimensions.csv"
    appearance_path = metadata_dir / "object_appearance.csv"
    objects = load_object_dimensions(dimensions_path)
    appearance_order = load_object_appearance(appearance_path)

    tasks = []
    tasks.extend(generate_measurement_tasks(objects, measurement))
    tasks.extend(generate_perspective_tasks(objects, perspective))
    tasks.extend(generate_spatiotemporal_tasks(appearance_order, spatiotemporal))

    return {
        "metadata_dir": str(metadata_dir),
        "task_count": len(tasks),
        "tasks": tasks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA tasks from trajectory metadata.")
    parser.add_argument("--metadata-dir", type=Path, required=True, help="Directory containing trajectory metadata CSVs.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the QA JSON.")
    parser.add_argument("--measurement-tasks", type=int, default=5, help="Number of measurement tasks to create.")
    parser.add_argument("--perspective-tasks", type=int, default=5, help="Number of perspective tasks to create.")
    parser.add_argument("--spatiotemporal-tasks", type=int, default=3, help="Number of spatiotemporal tasks to create.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    qa_payload = generate_tasks(
        metadata_dir=args.metadata_dir,
        measurement=args.measurement_tasks,
        perspective=args.perspective_tasks,
        spatiotemporal=args.spatiotemporal_tasks,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(qa_payload, f, indent=2)


if __name__ == "__main__":
    main()
