from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FAMILY_PATTERNS = [
    ("refrigerator", re.compile(r"refrigerator|fridge|freezer", re.I)),
    ("oven", re.compile(r"oven|microwave", re.I)),
    ("wardrobe", re.compile(r"wardrobe|closet", re.I)),
    ("cabinet", re.compile(r"cabinet|cupboard|bookcase|kitchenette", re.I)),
    ("drawer_storage", re.compile(
        r"drawer|dresser|nightstand|chest[_ -]?of[_ -]?drawers|desk", re.I
    )),
    ("appliance", re.compile(
        r"dishwasher|washing[_ -]?machine|washer|dryer", re.I
    )),
]
COMPONENT_PATTERN = re.compile(
    r"door|drawer|cabinet|lid|freezer|oven", re.I
)


def family_for(name: str) -> str | None:
    for family, pattern in FAMILY_PATTERNS:
        if pattern.search(name):
            return family
    return None


def parse_urdf(path: Path) -> dict[str, Any] | None:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    robot_name = root.attrib.get("name", path.stem)
    family = family_for(robot_name)
    if family is None:
        return None
    joints = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        name = joint.attrib.get("name", "")
        if (
            joint_type not in {"revolute", "prismatic", "continuous"}
            or not COMPONENT_PATTERN.search(name)
        ):
            continue
        limit = joint.find("limit")
        lower = float(limit.attrib.get("lower", "0")) if limit is not None else 0.0
        upper = float(limit.attrib.get("upper", "0")) if limit is not None else 0.0
        if joint_type != "continuous" and abs(upper - lower) <= 1e-6:
            continue
        joints.append(
            {
                "name": name,
                "joint_type": joint_type,
                "lower": lower,
                "upper": upper,
            }
        )
    if not joints:
        return None
    return {
        "template_name": path.stem,
        "robot_name": robot_name,
        "family": family,
        "eligible_components": joints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory HSSD articulated assets usable by Task C."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Habitat/data/scene_datasets/hssd-hab"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Habitat/hssd_taskc_articulated_inventory.json"),
    )
    args = parser.parse_args()
    root = args.dataset_root.resolve()

    templates: dict[str, dict[str, Any]] = {}
    for path in root.glob("urdf/*/*.urdf"):
        record = parse_urdf(path)
        if record is not None:
            templates[record["template_name"]] = record

    scene_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    scenes_by_family: dict[str, set[str]] = defaultdict(set)
    used_templates_by_family: dict[str, set[str]] = defaultdict(set)
    scene_records = []
    for path in sorted((root / "scenes-articulated").glob("*.scene_instance.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        matches = []
        for index, instance in enumerate(
            payload.get("articulated_object_instances", [])
        ):
            template_name = instance.get("template_name", "")
            template = templates.get(template_name)
            if template is None:
                continue
            family = template["family"]
            matches.append(
                {
                    "instance_index": index,
                    "template_name": template_name,
                    "robot_name": template["robot_name"],
                    "family": family,
                    "eligible_component_count": len(
                        template["eligible_components"]
                    ),
                }
            )
            instance_counts[family] += 1
            component_counts[family] += len(template["eligible_components"])
            scenes_by_family[family].add(path.stem)
            used_templates_by_family[family].add(template_name)
        for family in {match["family"] for match in matches}:
            scene_counts[family] += 1
        if matches:
            scene_records.append({"scene": path.stem, "instances": matches})

    summary = {
        family: {
            "scenes": len(scenes_by_family[family]),
            "instances": instance_counts[family],
            "eligible_components": component_counts[family],
            "unique_templates": len(used_templates_by_family[family]),
        }
        for family in sorted(instance_counts)
    }
    output = {
        "schema_version": "hssd-taskc-articulated-inventory-v1",
        "dataset_root": str(root),
        "articulated_scene_count": len(
            list((root / "scenes-articulated").glob("*.scene_instance.json"))
        ),
        "usable_scene_count": len(scene_records),
        "summary_by_family": summary,
        "templates": list(sorted(templates.values(), key=lambda row: (
            row["family"], row["robot_name"], row["template_name"]
        ))),
        "scenes": scene_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output": str(args.output),
            "usable_scene_count": len(scene_records),
            "summary_by_family": summary,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
