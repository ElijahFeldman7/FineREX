from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy import stats
import statsmodels.stats.multitest as mt
import numpy as np

FINETUNED_RUNS_DIR = Path("runs/splits")
BASELINE_RUNS_DIR = Path("runs/llama8b_splits")
OUTPUT_TXT = Path("scripts/statistics/wilks_results_8b.txt")
OUTPUT_JSON = Path("scripts/statistics/wilks_results_8b.json")

ENTITY_TYPES = [
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "MEANS_OF_TRANSPORTATION",
    "MEANS_OF_COMMUNICATION",
    "ROUTES",
    "SMUGGLED_ITEMS",
]

BOUND_SUFFIXES = {"precision", "recall", "f1"}


def corrected_resampled_ttest(diffs, n_train_percent=0.80, n_test_percent=0.10):
    n = len(diffs)
    mean_diffs = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)
    correction_factor = (1 / n) + (n_test_percent / n_train_percent)
    corrected_variance = var_diff * correction_factor
    t_stat = mean_diffs / np.sqrt(corrected_variance)
    df = n - 1
    p_val = 2 * stats.t.sf(abs(t_stat), df)
    return p_val


def harmonic_number(m: int) -> float:
    return float(sum(1.0 / i for i in range(1, m + 1)))


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class RunInfo:
    split_id: int
    metrics: Dict[str, float]
    timestamp: datetime


def load_runs(runs_dir: Path) -> Dict[int, RunInfo]:
    chosen: Dict[int, RunInfo] = {}
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        with meta_path.open("r") as f:
            meta = json.load(f)

        split_id = meta.get("split_id")
        metrics = meta.get("metrics")
        if split_id is None or metrics is None:
            continue

        ts = parse_time(meta.get("end_time"))
        if ts is None:
            ts = parse_time(meta.get("created_at"))
        if ts is None:
            ts = parse_time(meta.get("start_time"))
        if ts is None:
            ts = datetime.fromtimestamp(0)

        info = RunInfo(split_id=int(split_id), metrics=metrics, timestamp=ts)
        current = chosen.get(info.split_id)
        if current is None or info.timestamp > current.timestamp:
            chosen[info.split_id] = info

    return chosen


def metric_series(runs: Dict[int, RunInfo], splits: List[int], key: str) -> List[float]:
    values = []
    for split_id in splits:
        value = runs[split_id].metrics.get(key)
        if value is None:
            raise KeyError(f"Missing metric {key} for split {split_id}")
        values.append(float(value))
    return values


def logit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(values, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)),
        "n": int(len(arr)),
    }


def compute_tests(
    finetuned: List[float],
    baseline: List[float],
    metric_name: str,
) -> Dict[str, float]:
    ft = np.array(finetuned, dtype=float)
    base = np.array(baseline, dtype=float)

    use_logit = metric_name.endswith(tuple(BOUND_SUFFIXES))
    if use_logit:
        ft = logit(ft)
        base = logit(base)
        transform = "logit"
    else:
        transform = "none"

    diffs = ft - base
    _, shapiro_p = stats.shapiro(diffs)
    p_val = corrected_resampled_ttest(diffs)

    return {
        "p_raw": float(p_val),
        "shapiro_p": float(shapiro_p),
        "transform": transform,
    }


def main() -> None:
    finetuned_runs = load_runs(FINETUNED_RUNS_DIR)
    baseline_runs = load_runs(BASELINE_RUNS_DIR)

    splits = sorted(set(finetuned_runs.keys()) & set(baseline_runs.keys()))
    if not splits:
        raise RuntimeError("No overlapping splits found between finetuned and baseline runs.")

    global_metrics = [
        "test_entity_precision",
        "test_entity_recall",
        "test_entity_f1",
        "test_relationship_precision",
        "test_relationship_recall",
        "test_relationship_f1",
        "test_relationship_score_mae",
    ]

    per_type_metrics = []
    for entity_type in ENTITY_TYPES:
        per_type_metrics.extend([
            f"test_entity_{entity_type}_precision",
            f"test_entity_{entity_type}_recall",
            f"test_entity_{entity_type}_f1",
        ])

    metric_keys = global_metrics + per_type_metrics
    m = len(metric_keys)
    harmonic_m = harmonic_number(m)
    by_factor = m * harmonic_m

    results = {}
    p_values = []

    for metric in metric_keys:
        ft_values = metric_series(finetuned_runs, splits, metric)
        base_values = metric_series(baseline_runs, splits, metric)
        tests = compute_tests(ft_values, base_values, metric)
        stats_ft = compute_stats(ft_values)
        stats_base = compute_stats(base_values)
        results[metric] = {
            "finetuned": stats_ft,
            "baseline": stats_base,
            **tests,
        }
        p_values.append(tests["p_raw"])

    reject, corrected_p_values, _, _ = mt.multipletests(p_values, alpha=0.01, method="fdr_by")
    for metric, corrected_p, decision in zip(metric_keys, corrected_p_values, reject):
        results[metric]["p_corrected"] = float(corrected_p)
        results[metric]["reject_null"] = bool(decision)

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TXT.open("w") as f:
        f.write("Wilks results (80/10/10 corrected resampled t-test)\n")
        f.write(f"Splits used: {splits}\n")
        f.write(f"Total hypotheses: {m}\n")
        f.write(f"Harmonic H_m: {harmonic_m:.12f}\n")
        f.write(f"BY factor (m*H_m): {by_factor:.12f}\n\n")
        for metric in metric_keys:
            row = results[metric]
            f.write(f"{metric}\n")
            f.write(
                f"  finetuned mean={row['finetuned']['mean']:.6f} sd={row['finetuned']['sd']:.6f}\n"
            )
            f.write(
                f"  baseline mean={row['baseline']['mean']:.6f} sd={row['baseline']['sd']:.6f}\n"
            )
            f.write(
                f"  shapiro_p={row['shapiro_p']:.6f} transform={row['transform']}\n"
            )
            f.write(
                f"  p_raw={row['p_raw']:.6e} p_corrected={row['p_corrected']:.6e} reject={row['reject_null']}\n\n"
            )

    with OUTPUT_JSON.open("w") as f:
        json.dump(
            {
                "splits": splits,
                "metric_keys": metric_keys,
                "by_correction": {
                    "m": m,
                    "harmonic_m": harmonic_m,
                    "factor": by_factor,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Wrote {OUTPUT_TXT}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()