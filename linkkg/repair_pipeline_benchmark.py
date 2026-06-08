from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
LINKKG_DIR = REPO_ROOT / "LinkKG-HS" / "linkkg"
PROMPTS_DIR = LINKKG_DIR / "prompts"
RUNS_ROOT = REPO_ROOT / "runs" / "pipeline_benchmark"
GRAPHRAG_DIR = LINKKG_DIR / "kgconstruction"
GRAPHRAG_DEFAULT_CONFIG = GRAPHRAG_DIR / "ragtest" / "settings.yaml"

ENTITY_CONFIGS = {
    "person": "person_nopr",
    "location": "location_nopr",
    "organization": "org_nopr",
    "routes": "routes_nopr",
    "means_of_transportation": "mot_nopr",
    "means_of_communication": "moc_nopr",
    "smuggled_items": "smuggleditems_nopr",
}


def entity_prompt_paths(entity_type: str) -> Dict[str, Path]:
    prefix = ENTITY_CONFIGS[entity_type]
    return {
        "ner": PROMPTS_DIR / f"{prefix}_ner_prompt.txt",
        "coref": PROMPTS_DIR / f"{prefix}_coref_prompt.txt",
        "resolve": PROMPTS_DIR / f"{prefix}_resolve_prompt.txt",
    }


def write_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_cmd(cmd: List[str], cwd: Path, timing_path: Path, meta: Dict, dry_run: bool) -> None:
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return

    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    end = time.time()

    record = {
        **meta,
        "start_time": datetime.fromtimestamp(start).isoformat(),
        "end_time": datetime.fromtimestamp(end).isoformat(),
        "duration_seconds": round(end - start, 2),
        "returncode": result.returncode,
    }
    write_jsonl(timing_path, record)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def dir_has_files(path: Path, pattern: str) -> bool:
    return path.exists() and any(path.glob(pattern))


def has_prep_outputs(output_dir: Path) -> bool:
    return dir_has_files(output_dir / "chunk_outputs", "chunk_*.txt") and dir_has_files(
        output_dir / "ner_outputs", "*.json"
    )


def has_ner_outputs(output_dir: Path) -> bool:
    return dir_has_files(output_dir / "ner_outputs", "*.json")


def has_coref_outputs(output_dir: Path) -> bool:
    return (output_dir / "final_memory.json").exists()


def has_resolved_output(output_dir: Path, case_name: str, entity_type: str) -> bool:
    return (output_dir / f"{entity_type}_resolved_{case_name}.txt").exists()


def has_kg_output(run_root: Path, approach: str, case_name: str) -> bool:
    kg_dir = run_root / "kgs" / approach / case_name
    return (kg_dir / "kg_stats.json").exists() or (kg_dir / f"{case_name}_kg.graphml").exists()


def has_graphrag_output(project_root: Path) -> bool:
    output_root = project_root / "output"
    if not output_root.exists():
        return False
    return any(p.is_dir() for p in output_root.iterdir())


def write_latest_graphrag_outputs(project_root: Path) -> None:
    output_root = project_root / "output"
    if not output_root.exists():
        return

    candidates = [p for p in output_root.iterdir() if p.is_dir()]
    if not candidates:
        return

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    (project_root / "LATEST.txt").write_text(str(latest), encoding="utf-8")

    graph_files = sorted(latest.rglob("*.graphml"), key=lambda p: p.stat().st_mtime)
    if graph_files:
        shutil.copy2(graph_files[-1], project_root / "graph_latest.graphml")

    image_files = sorted(latest.rglob("*.png"), key=lambda p: p.stat().st_mtime)
    if image_files:
        shutil.copy2(image_files[-1], project_root / "graph_latest.png")


def load_cases(run_root: Path, filter_case: str | None) -> Iterable[str]:
    if filter_case:
        return [filter_case]
    timing_dir = run_root / "timing"
    return [p.stem for p in timing_dir.glob("*.jsonl")]


def run_full_70b_entity(
    run_root: Path,
    case_name: str,
    entity_type: str,
    model_name: str,
    timing_path: Path,
    dry_run: bool,
    force: bool,
) -> None:
    chunk_dir = run_root / "common" / case_name / "chunk_outputs"
    output_dir = run_root / "full_70b" / case_name / entity_type
    log_file = output_dir / "log.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = entity_prompt_paths(entity_type)

    if force or not has_ner_outputs(output_dir):
        ner_cmd = [
            sys.executable,
            "ner.py",
            "--chunks-dir",
            str(chunk_dir),
            "--prompt-file",
            str(paths["ner"]),
            "--output-dir",
            str(output_dir / "ner_outputs"),
            "--log-file",
            str(log_file),
            "--model-name",
            model_name,
            "--max-retries",
            "2",
        ]
        run_cmd(ner_cmd, LINKKG_DIR, timing_path, {
            "stage": "ner",
            "case": case_name,
            "entity_type": entity_type,
            "approach": "full_70b",
            "model": model_name,
        }, dry_run)

    if force or not has_coref_outputs(output_dir):
        coref_cmd = [
            sys.executable,
            "loopcoref.py",
            "--chunks-dir",
            str(chunk_dir),
            "--ner-dir",
            str(output_dir / "ner_outputs"),
            "--prompt-file",
            str(paths["coref"]),
            "--base-output-folder",
            str(output_dir),
            "--input-file-name",
            case_name,
            "--model",
            model_name,
            "--verify-passes",
            "0",
            "--log-file",
            str(log_file),
            "--max-retries",
            "3",
        ]
        run_cmd(coref_cmd, LINKKG_DIR, timing_path, {
            "stage": "coref",
            "case": case_name,
            "entity_type": entity_type,
            "approach": "full_70b",
            "model": model_name,
        }, dry_run)

    if force or not has_resolved_output(output_dir, case_name, entity_type):
        resolve_cmd = [
            sys.executable,
            "resolve_coref.py",
            "--chunks-dir",
            str(chunk_dir),
            "--final-memory",
            str(output_dir / "final_memory.json"),
            "--prompt-file",
            str(paths["resolve"]),
            "--base-output-dir",
            str(output_dir),
            "--input-file-name",
            case_name,
            "--model-name",
            model_name,
            "--num-retries",
            "1",
            "--num-ctx",
            "8192",
            "--request-timeout",
            "600",
            "--log-file",
            str(log_file),
            "--entity-type",
            entity_type,
        ]
        run_cmd(resolve_cmd, LINKKG_DIR, timing_path, {
            "stage": "resolve",
            "case": case_name,
            "entity_type": entity_type,
            "approach": "full_70b",
            "model": model_name,
        }, dry_run)


def run_full_finetuned_entity(
    run_root: Path,
    case_name: str,
    entity_type: str,
    model_name: str,
    timing_path: Path,
    dry_run: bool,
    force: bool,
) -> None:
    output_dir = run_root / "full_finetuned" / case_name / entity_type
    log_file = output_dir / "log.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_csv = run_root / "nerre_extractions" / case_name / "extraction.csv"
    if not extraction_csv.exists():
        raise FileNotFoundError(f"Missing extraction.csv: {extraction_csv}")

    paths = entity_prompt_paths(entity_type)

    if force or not has_prep_outputs(output_dir):
        prep_cmd = [
            sys.executable,
            "prep_from_csv.py",
            "--csv-file",
            str(extraction_csv),
            "--output-dir",
            str(output_dir),
            "--entity-type",
            entity_type,
        ]
        run_cmd(prep_cmd, LINKKG_DIR, timing_path, {
            "stage": "prep",
            "case": case_name,
            "entity_type": entity_type,
            "approach": "full_finetuned",
        }, dry_run)

    if force or not has_coref_outputs(output_dir):
        coref_cmd = [
            sys.executable,
            "loopcoref.py",
            "--chunks-dir",
            str(output_dir / "chunk_outputs"),
            "--ner-dir",
            str(output_dir / "ner_outputs"),
            "--prompt-file",
            str(paths["coref"]),
            "--base-output-folder",
            str(output_dir),
            "--input-file-name",
            case_name,
            "--model",
            model_name,
            "--verify-passes",
            "0",
            "--log-file",
            str(log_file),
            "--max-retries",
            "3",
        ]
        run_cmd(coref_cmd, LINKKG_DIR, timing_path, {
            "stage": "coref",
            "case": case_name,
            "entity_type": entity_type,
            "approach": "full_finetuned",
            "model": model_name,
        }, dry_run)

    if force or not has_resolved_output(output_dir, case_name, entity_type):
        resolve_cmd = [
            sys.executable,
            "resolve_coref.py",
            "--chunks-dir",
            str(output_dir / "chunk_outputs"),
            "--final-memory",
            str(output_dir / "final_memory.json"),
            "--prompt-file",
            str(paths["resolve"]),
            "--base-output-dir",
            str(output_dir),
            "--input-file-name",
            case_name,
            "--model-name",
            model_name,
            "--num-retries",
            "1",
            "--num-ctx",
            "8192",
            "--request-timeout",
            "600",
            "--log-file",
            str(log_file),
            "--entity-type",
            entity_type,
        ]
        run_cmd(resolve_cmd, LINKKG_DIR, timing_path, {
            "stage": "resolve",
            "case": case_name,
            "entity_type": entity_type,
            "approach": "full_finetuned",
            "model": model_name,
        }, dry_run)


def run_kg(run_root: Path, approach: str, case_name: str, timing_path: Path, dry_run: bool) -> None:
    from generate_kgs import create_consolidated_kg, visualize_kg, save_kg_as_graphml, save_kg_stats

    kg_dir = run_root / "kgs" / approach / case_name
    kg_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"DRY RUN: generate KG for {approach}/{case_name}")
        return

    kg_start = time.time()
    G, entities_by_type = create_consolidated_kg(str(run_root / approach), case_name, str(kg_dir))
    if G.number_of_nodes() > 0:
        visualize_kg(G, str(kg_dir / f"{case_name}_kg.png"), case_name)
        save_kg_as_graphml(G, str(kg_dir / f"{case_name}_kg.graphml"))
        save_kg_stats(G, str(kg_dir / "kg_stats.json"), entities_by_type)
    kg_end = time.time()

    write_jsonl(timing_path, {
        "stage": "kg",
        "case": case_name,
        "entity_type": "all",
        "approach": approach,
        "start_time": datetime.fromtimestamp(kg_start).isoformat(),
        "end_time": datetime.fromtimestamp(kg_end).isoformat(),
        "duration_seconds": round(kg_end - kg_start, 2),
    })


def run_graphrag_smuggled_items(
    run_root: Path,
    approach: str,
    case_name: str,
    graphrag_root: Path,
    config_path: Path,
    timing_path: Path,
    dry_run: bool,
    force: bool,
) -> None:
    resolved_path = (
        run_root
        / approach
        / case_name
        / "smuggled_items"
        / f"smuggled_items_resolved_{case_name}.txt"
    )
    if not resolved_path.exists():
        print(f"Skip GraphRAG: missing {resolved_path}")
        return

    project_root = graphrag_root / approach / case_name
    input_dir = project_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Missing GraphRAG config: {config_path}")

    project_config_path = project_root / "settings.yaml"
    if force or not project_config_path.exists() or config_path.stat().st_mtime > project_config_path.stat().st_mtime:
        shutil.copy2(config_path, project_config_path)

    input_file = input_dir / f"{case_name}_smuggled_items_resolved.txt"
    if force or not input_file.exists() or resolved_path.stat().st_mtime > input_file.stat().st_mtime:
        shutil.copy2(resolved_path, input_file)

    if not force and has_graphrag_output(project_root):
        return

    cmd = [
        sys.executable,
        str(GRAPHRAG_DIR / "index.py"),
        "--root",
        str(project_root),
    ]
    run_cmd(cmd, GRAPHRAG_DIR, timing_path, {
        "stage": "graphrag",
        "case": case_name,
        "entity_type": "smuggled_items",
        "approach": f"graphrag_{approach}",
    }, dry_run)

    if not dry_run:
        write_latest_graphrag_outputs(project_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume missing benchmark stages")
    parser.add_argument("--run-id", required=True, help="pipeline_benchmark run id (e.g., 7419659)")
    parser.add_argument("--case", default=None, help="case name to repair (default: all in timing)")
    parser.add_argument("--approach", nargs="+", default=["full_70b", "full_finetuned"])
    parser.add_argument("--model-70b", default="llama3.1:70b")
    parser.add_argument("--graphrag-root", default=None)
    parser.add_argument("--skip-graphrag", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun even if outputs exist")
    args = parser.parse_args()

    run_root = RUNS_ROOT / args.run_id
    if not run_root.exists():
        raise FileNotFoundError(f"Run not found: {run_root}")

    cases = load_cases(run_root, args.case)
    graphrag_root = Path(args.graphrag_root) if args.graphrag_root else run_root / "graphrag"
    graphrag_config = GRAPHRAG_DEFAULT_CONFIG
    run_graphrag = not args.skip_graphrag

    for case_name in cases:
        timing_path = run_root / "timing" / f"{case_name}.jsonl"
        for approach in args.approach:
            if approach == "full_70b":
                for entity_type in ENTITY_CONFIGS.keys():
                    run_full_70b_entity(
                        run_root,
                        case_name,
                        entity_type,
                        args.model_70b,
                        timing_path,
                        args.dry_run,
                        args.force,
                    )
                if args.force or not has_kg_output(run_root, approach, case_name):
                    run_kg(run_root, approach, case_name, timing_path, args.dry_run)
            elif approach == "full_finetuned":
                for entity_type in ENTITY_CONFIGS.keys():
                    run_full_finetuned_entity(
                        run_root,
                        case_name,
                        entity_type,
                        args.model_70b,
                        timing_path,
                        args.dry_run,
                        args.force,
                    )
                if args.force or not has_kg_output(run_root, approach, case_name):
                    run_kg(run_root, approach, case_name, timing_path, args.dry_run)
            else:
                raise ValueError(f"Unknown approach: {approach}")

            if run_graphrag:
                run_graphrag_smuggled_items(
                    run_root,
                    approach,
                    case_name,
                    graphrag_root,
                    graphrag_config,
                    timing_path,
                    args.dry_run,
                    args.force,
                )


if __name__ == "__main__":
    main()
