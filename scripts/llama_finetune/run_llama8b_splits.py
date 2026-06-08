import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from transformers import AutoTokenizer

from .config import DATASET_PATH, format_system_prompt
from .metrics import compute_metrics
from .trainer import normalize_extraction

INSTRUCTION_TEMPLATE = """Input_text: \n{input_text}\nOutput:\n"""

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
_OLLAMA_HOST = os.getenv("OLLAMA_HOST")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL")
if DEFAULT_OLLAMA_URL is None:
    if _OLLAMA_HOST:
        DEFAULT_OLLAMA_URL = f"http://{_OLLAMA_HOST}/api/generate"
    else:
        DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 2056
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TIMEOUT = 600
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 2


@dataclass
class RunInfo:
    split_id: int
    run_dir: Path
    end_time: Optional[datetime]
    system_prompt: Optional[str]


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
    if split_id is None:
        return None

    end_time = parse_time(meta.get("end_time"))
    system_prompt = meta.get("system_prompt")
    return RunInfo(split_id=int(split_id), run_dir=run_dir, end_time=end_time, system_prompt=system_prompt)


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


def get_completed_splits(output_root: Path) -> Set[int]:
    completed: Set[int] = set()
    if not output_root.exists():
        return completed

    for run_dir in output_root.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        try:
            with meta_path.open("r") as f:
                meta = json.load(f)
            split_id = meta.get("split_id")
            if split_id is not None:
                completed.add(int(split_id))
        except (json.JSONDecodeError, OSError, ValueError):
            continue

    return completed


def run_ollama_request(
    prompt: str,
    model: str,
    api_url: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout_seconds: int,
    retries: int,
    backoff_seconds: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_new_tokens,
        },
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_exc: Exception = RuntimeError("Unknown Ollama request error")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            output = body.get("response", "")
            if not isinstance(output, str):
                raise RuntimeError(f"Unexpected Ollama response format: {body}")
            return output.strip()
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            wait_seconds = backoff_seconds * attempt
            print(
                f"    ! Ollama request failed (attempt {attempt}/{retries}): {exc}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to call Ollama API after {retries} attempts: {last_exc}"
    )


def clean_ground_truth(text: str) -> str:
    if not isinstance(text, str):
        return ""
    gt = text.strip()
    gt = gt.replace("{tuple_delimiter}", "|")
    gt = gt.replace("{record_delimiter}", "\n")
    gt = gt.replace("{completion_delimiter}", "<END>")
    return gt


def load_test_rows(run_dir: Path, dataset_path: Path) -> List[Dict[str, str]]:
    test_csv = run_dir / "test_split.csv"
    if test_csv.exists():
        rows = []
        with test_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    manifest_path = run_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing test split and manifest in {run_dir}")

    with manifest_path.open("r") as f:
        manifest = json.load(f)
    test_ids = set(manifest.get("test_row_ids", []))

    df = pd.read_csv(dataset_path).fillna("")
    df["row_id"] = df.index
    df = df[df["row_id"].isin(test_ids)]

    rows = []
    for row in df.itertuples(index=False):
        rows.append({
            "row_id": str(row.row_id),
            "Input_Text": row.Input_Text,
            "Output": row.Output,
        })
    return rows


def write_summary_report(
    output_path: Path,
    global_metrics: Dict[str, float],
    per_sample_blocks: List[str],
) -> None:
    summary_report_header = "===== GLOBAL SUMMARY =====\n"
    summary_report_header += (
        f"Entities: TP={global_metrics.get('entity_tp', 0)}, "
        f"FP={global_metrics.get('entity_fp', 0)}, "
        f"FN={global_metrics.get('entity_fn', 0)}\n"
    )
    summary_report_header += (
        f"P={global_metrics.get('entity_precision', 0):.4f}, "
        f"R={global_metrics.get('entity_recall', 0):.4f}, "
        f"F1={global_metrics.get('entity_f1', 0):.4f}\n\n"
    )

    summary_report_header += (
        f"Relations: TP={global_metrics.get('relationship_tp', 0)}, "
        f"FP={global_metrics.get('relationship_fp', 0)}, "
        f"FN={global_metrics.get('relationship_fn', 0)}\n"
    )
    summary_report_header += (
        f"P={global_metrics.get('relationship_precision', 0):.4f}, "
        f"R={global_metrics.get('relationship_recall', 0):.4f}, "
        f"F1={global_metrics.get('relationship_f1', 0):.4f}\n\n"
    )

    summary_report_header += "===== PER-TYPE ENTITY SUMMARY =====\n\n"
    entity_types = [
        "PERSON",
        "LOCATION",
        "ORGANIZATION",
        "MEANS_OF_TRANSPORTATION",
        "MEANS_OF_COMMUNICATION",
        "ROUTES",
        "SMUGGLED_ITEMS",
    ]
    for etype in entity_types:
        tp = global_metrics.get(f"entity_{etype}_tp", 0)
        fp = global_metrics.get(f"entity_{etype}_fp", 0)
        fn = global_metrics.get(f"entity_{etype}_fn", 0)
        p = global_metrics.get(f"entity_{etype}_precision", 0)
        r = global_metrics.get(f"entity_{etype}_recall", 0)
        f1 = global_metrics.get(f"entity_{etype}_f1", 0)

        summary_report_header += f"{etype}:\n"
        summary_report_header += f"  TP={tp}, FP={fp}, FN={fn}\n"
        summary_report_header += f"  P={p:.4f}, R={r:.4f}, F1={f1:.4f}\n\n"

    summary_report_header += "--- GLOBAL METRICS (AVERAGE) ---\n"
    summary_report_header += json.dumps(global_metrics, indent=2) + "\n"
    summary_report_header += "=" * 30 + "\n\n"

    summary_report = summary_report_header + "".join(per_sample_blocks)
    output_path.write_text(summary_report)


def process_split(
    run_info: RunInfo,
    output_root: Path,
    ollama_model: str,
    ollama_url: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer_name: str,
    timeout_seconds: int,
    retries: int,
    backoff_seconds: int,
) -> None:
    dataset_path = Path(DATASET_PATH)
    test_rows = load_test_rows(run_info.run_dir, dataset_path)

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"llama8b_split{run_info.split_id:02d}_{run_stamp}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = run_info.system_prompt or format_system_prompt()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    predictions_rows: List[Dict[str, str]] = []
    jsonl_rows: List[Dict[str, str]] = []
    detailed_rows: List[Dict[str, str]] = []
    per_sample_blocks: List[str] = []

    decoded_predictions: List[str] = []
    decoded_ground_truths: List[str] = []

    for idx, row in enumerate(test_rows):
        input_text = row.get("Input_Text", "")
        raw_gt = row.get("Output", "")
        row_id = row.get("row_id")

        user_prompt = INSTRUCTION_TEMPLATE.format(input_text=input_text).strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output_text = run_ollama_request(
            prompt=prompt_str,
            model=ollama_model,
            api_url=ollama_url,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )

        if "<END>" in output_text:
            output_text = output_text.split("<END>")[0] + "<END>"

        gt = clean_ground_truth(raw_gt)

        pred_norm = normalize_extraction(output_text)
        gt_norm = normalize_extraction(gt)

        decoded_predictions.append(pred_norm)
        decoded_ground_truths.append(gt_norm)

        predictions_rows.append({
            "row_id": row_id,
            "Input_Text": input_text,
            "Ground_Truth": gt,
            "Predicted_Text": output_text,
        })

        sample_metrics = compute_metrics([pred_norm], [gt_norm])
        per_sample_blocks.append(
            f"--- Sample {idx} ---\n"
            f"Input Data: {input_text}\n\n"
            f"Prediction:\n{pred_norm}\n"
            f"Ground Truth:\n{gt_norm}\n"
            f"Metrics: {json.dumps(sample_metrics, indent=2)}\n\n"
        )

        jsonl_rows.append({
            "sample_id": idx,
            "row_id": row_id,
            "prompt": input_text,
            "prediction": pred_norm,
            "ground_truth": gt_norm,
            "metrics": sample_metrics,
        })

        detailed_rows.append({
            "Row_ID": idx + 1,
            "TP_entities": sample_metrics.get("entity_tp", 0),
            "FP_entities": sample_metrics.get("entity_fp", 0),
            "FN_entities": sample_metrics.get("entity_fn", 0),
            "TP_rel": sample_metrics.get("relationship_tp", 0),
            "FP_rel": sample_metrics.get("relationship_fp", 0),
            "FN_rel": sample_metrics.get("relationship_fn", 0),
            "TP_entity_pairs": sample_metrics.get("tp_entity_pairs", ""),
            "FP_entity_pairs": sample_metrics.get("fp_entity_pairs", ""),
            "FN_entity_pairs": sample_metrics.get("fn_entity_pairs", ""),
            "TP_relation_pairs": sample_metrics.get("tp_relation_pairs", ""),
            "FP_relation_pairs": sample_metrics.get("fp_relation_pairs", ""),
            "FN_relation_pairs": sample_metrics.get("fn_relation_pairs", ""),
        })

    global_metrics = compute_metrics(decoded_predictions, decoded_ground_truths)

    predictions_path = run_dir / "predictions.csv"
    with predictions_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["row_id", "Input_Text", "Ground_Truth", "Predicted_Text"],
        )
        writer.writeheader()
        writer.writerows(predictions_rows)

    jsonl_path = run_dir / "metrics.jsonl"
    with jsonl_path.open("w") as f:
        for entry in jsonl_rows:
            f.write(json.dumps(entry) + "\n")

    detailed_path = run_dir / "detailed_metrics.csv"
    with detailed_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(detailed_rows[0].keys()) if detailed_rows else [],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_path = run_dir / "summary_report.txt"
    write_summary_report(summary_path, global_metrics, per_sample_blocks)

    metrics_payload = {f"test_{k}": v for k, v in global_metrics.items()}
    meta_payload = {
        "run_id": run_id,
        "split_id": run_info.split_id,
        "source_run_dir": str(run_info.run_dir),
        "model": ollama_model,
        "ollama_url": ollama_url,
        "tokenizer": tokenizer_name,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "system_prompt": system_prompt,
        "metrics": metrics_payload,
        "created_at": datetime.now().isoformat(),
    }

    with (run_dir / "run_meta.json").open("w") as f:
        json.dump(meta_payload, f, indent=2)

    print(f"[split {run_info.split_id}] Wrote outputs to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Llama 8B on test splits")
    parser.add_argument("--runs-dir", default="runs/splits")
    parser.add_argument("--output-root", default="runs/llama8b_splits")
    parser.add_argument("--split-id", type=int, default=None)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff", type=int, default=DEFAULT_BACKOFF)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--list-missing", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    runs, _ = pick_latest_runs(runs_dir)
    if args.split_id is not None:
        runs = [r for r in runs if r.split_id == args.split_id]
        if not runs:
            raise ValueError(f"Split {args.split_id} not found under {runs_dir}")

    runs = sorted(runs, key=lambda r: r.split_id)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    completed = get_completed_splits(output_root)
    if args.list_missing:
        missing = [r.split_id for r in runs if r.split_id not in completed]
        print(",".join(str(x) for x in missing))
        return

    if args.skip_existing:
        before = len(runs)
        runs = [r for r in runs if r.split_id not in completed]
        skipped = before - len(runs)
        if skipped:
            print(f"Skipping {skipped} split(s) with existing outputs in {output_root}")

    for run_info in runs:
        process_split(
            run_info=run_info,
            output_root=output_root,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer_name=args.tokenizer_name,
            timeout_seconds=args.timeout,
            retries=args.retries,
            backoff_seconds=args.backoff,
        )


if __name__ == "__main__":
    main()
