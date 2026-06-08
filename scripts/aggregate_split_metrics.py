import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RunInfo:
    split_id: int
    run_dir: Path
    end_time: Optional[datetime]
    metrics: Dict[str, float]


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_run_meta(run_dir: Path) -> Optional[RunInfo]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return None

    with meta_path.open("r") as f:
        meta = json.load(f)

    split_id = meta.get("split_id")
    metrics = meta.get("metrics")
    if split_id is None or metrics is None:
        return None

    end_time = parse_time(meta.get("end_time"))
    return RunInfo(split_id=int(split_id), run_dir=run_dir, end_time=end_time, metrics=metrics)


def pick_latest_runs(runs_dir: Path) -> Tuple[List[RunInfo], List[str]]:
    chosen: Dict[int, RunInfo] = {}
    notes: List[str] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        info = load_run_meta(run_dir)
        if info is None:
            continue

        current = chosen.get(info.split_id)
        if current is None:
            chosen[info.split_id] = info
            continue

        current_time = current.end_time or datetime.fromtimestamp(current.run_dir.stat().st_mtime)
        new_time = info.end_time or datetime.fromtimestamp(info.run_dir.stat().st_mtime)

        if new_time > current_time:
            notes.append(
                f"Split {info.split_id}: replaced {current.run_dir.name} with {info.run_dir.name}"
            )
            chosen[info.split_id] = info
        else:
            notes.append(
                f"Split {info.split_id}: kept {current.run_dir.name} over {info.run_dir.name}"
            )

    return list(chosen.values()), notes


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    if arr.size == 1:
        return {"mean": float(arr.mean()), "sd": 0.0, "n": int(arr.size)}
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)), "n": int(arr.size)}


def find_predictions_file(run_dir: Path, metrics: Dict[str, float]) -> Optional[Path]:
    epoch_value = metrics.get("epoch") if metrics else None
    epoch_label = None
    if isinstance(epoch_value, (int, float)):
        epoch_label = int(epoch_value)

    if epoch_label is not None:
        candidate = run_dir / f"predictions_epoch_{epoch_label}_with_row_id.csv"
        if candidate.exists():
            return candidate
        candidate = run_dir / f"predictions_epoch_{epoch_label}.csv"
        if candidate.exists():
            return candidate

    pattern = re.compile(r"predictions_epoch_(\d+)(?:_with_row_id)?\.csv")
    candidates = {}
    for path in run_dir.glob("predictions_epoch_*.csv"):
        match = pattern.match(path.name)
        if match:
            epoch = int(match.group(1))
            existing = candidates.get(epoch)
            if existing is None or path.name.endswith("_with_row_id.csv"):
                candidates[epoch] = path

    if not candidates:
        return None

    latest_epoch = max(candidates.keys())
    return candidates[latest_epoch]


def parse_ground_truth_entities(text: str) -> Dict[str, int]:
    if not isinstance(text, str):
        return {}

    cleaned = text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
    cleaned = cleaned.replace("<end>", "").strip()

    d = re.escape("|")
    entity_pattern = re.compile(
        r'\(\s*"entity"\s*' + d + r'\s*(.*?)\s*' + d + r'\s*(.*?)\s*' + d + r'.*?\)',
        re.DOTALL,
    )

    entities = set()
    for name, entity_type in entity_pattern.findall(cleaned):
        entities.add((name.strip().upper(), entity_type.strip().upper()))

    counts: Dict[str, int] = defaultdict(int)
    for _, entity_type in entities:
        counts[entity_type] += 1
    return counts


def compute_gt_type_averages(predictions_path: Path) -> Tuple[Dict[str, float], int]:
    sums: Dict[str, int] = defaultdict(int)
    total_samples = 0

    with predictions_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_text = (
                row.get("Ground_Truth")
                or row.get("Ground Truth")
                or row.get("ground_truth")
                or ""
            )
            counts = parse_ground_truth_entities(gt_text)
            for entity_type, count in counts.items():
                sums[entity_type] += count
            total_samples += 1

    if total_samples == 0:
        return {}, 0

    averages = {entity_type: count / total_samples for entity_type, count in sums.items()}
    return averages, total_samples


def aggregate_metrics(runs: List[RunInfo], notes: List[str]) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    summary: Dict[str, dict] = {}

    overall_keys = {
        "entity": {
            "precision": "test_entity_precision",
            "recall": "test_entity_recall",
            "f1": "test_entity_f1",
        },
        "relationship": {
            "precision": "test_relationship_precision",
            "recall": "test_relationship_recall",
            "f1": "test_relationship_f1",
        },
        "quality": {
            "parsability": "test_parsability_score",
            "relationship_score_mae": "test_relationship_score_mae",
        },
    }

    for scope, metrics_map in overall_keys.items():
        for label, key in metrics_map.items():
            values = [r.metrics[key] for r in runs if key in r.metrics]
            stats = compute_stats(values)
            rows.append({"scope": scope, "metric": label, **stats})
            summary[f"{scope}_{label}"] = stats

    type_pattern = re.compile(r"^test_entity_(.+)_precision$")
    entity_types = set()
    for r in runs:
        for key in r.metrics.keys():
            match = type_pattern.match(key)
            if match:
                entity_types.add(match.group(1))

    for entity_type in sorted(entity_types):
        for label in ("precision", "recall", "f1"):
            key = f"test_entity_{entity_type}_{label}"
            values = [r.metrics[key] for r in runs if key in r.metrics]
            stats = compute_stats(values)
            rows.append({"scope": f"entity_{entity_type}", "metric": label, **stats})
            summary[f"entity_{entity_type}_{label}"] = stats

    gt_averages_by_run: List[Dict[str, float]] = []
    for run in runs:
        predictions_path = find_predictions_file(run.run_dir, run.metrics)
        if not predictions_path:
            notes.append(f"Split {run.split_id}: missing predictions file")
            continue

        averages, sample_count = compute_gt_type_averages(predictions_path)
        if sample_count == 0:
            notes.append(f"Split {run.split_id}: empty predictions file")
            continue
        gt_averages_by_run.append(averages)

    if gt_averages_by_run:
        gt_types = set()
        for averages in gt_averages_by_run:
            gt_types.update(averages.keys())

        for entity_type in sorted(gt_types):
            values = [avg.get(entity_type, 0.0) for avg in gt_averages_by_run]
            stats = compute_stats(values)
            rows.append({"scope": f"gt_entity_{entity_type}", "metric": "avg_count", **stats})
            summary[f"gt_entity_{entity_type}_avg_count"] = stats
    else:
        notes.append("No ground truth entity averages computed.")

    return rows, summary


def write_outputs(output_dir: Path, rows: List[dict], summary: dict, runs: List[RunInfo], notes: List[str]):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "aggregate_metrics.csv"
    with csv_path.open("w") as f:
        f.write("scope,metric,mean,sd,n\n")
        for row in rows:
            f.write(
                f"{row['scope']},{row['metric']},{row['mean']:.6f},{row['sd']:.6f},{row['n']}\n"
            )

    json_path = output_dir / "aggregate_metrics.json"
    payload = {
        "runs_dir": str(output_dir),
        "run_count": len(runs),
        "runs": [
            {"split_id": r.split_id, "run_dir": str(r.run_dir), "end_time": r.end_time.isoformat() if r.end_time else None}
            for r in runs
        ],
        "summary": summary,
        "notes": notes,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate split metrics")
    parser.add_argument("--runs-dir", default="runs/splits")
    parser.add_argument("--output-dir", default="runs/splits")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    runs, notes = pick_latest_runs(runs_dir)
    runs = sorted(runs, key=lambda r: r.split_id)

    rows, summary = aggregate_metrics(runs, notes)

    print(f"Selected {len(runs)} runs from {runs_dir}")
    if len(runs) != 30:
        print("WARNING: Expected 30 runs. Check missing splits or duplicates.")

    write_outputs(Path(args.output_dir), rows, summary, runs, notes)


if __name__ == "__main__":
    main()
