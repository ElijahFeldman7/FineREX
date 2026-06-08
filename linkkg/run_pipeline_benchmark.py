import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import urllib.error
import urllib.request

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
LINKKG_DIR = REPO_ROOT / "LinkKG-HS" / "linkkg"
PROMPTS_DIR = LINKKG_DIR / "prompts"
INPUT_DIR = REPO_ROOT / "datasets" / "cleanedinput"

ENTITY_CONFIGS = {
    "person": "person_nopr",
    "location": "location_nopr",
    "organization": "org_nopr",
    "routes": "routes_nopr",
    "means_of_transportation": "mot_nopr",
    "means_of_communication": "moc_nopr",
    "smuggled_items": "smuggleditems_nopr",
}

DEFAULT_APPROACHES = ["full_70b", "full_finetuned", "shortcut"]

DEFAULT_NERRE_MODEL = "2028efeldman/llama-finetuned:latest"
DEFAULT_70B_MODEL = "llama3.1:70b"
DEFAULT_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 4000
DEFAULT_CHUNK_TOKENS = 300
DEFAULT_TIMEOUT = 600
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 2

INSTRUCTION_TEMPLATE = """Input_text: \n{input_text}\nOutput:\n"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def record_stage(timing_path: Path, payload: Dict) -> None:
    write_jsonl(timing_path, payload)


def run_cmd(cmd: List[str], cwd: Path, timing_path: Path, meta: Dict) -> None:
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
    record_stage(timing_path, record)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def load_system_prompt() -> str:
    sys.path.append(str(REPO_ROOT))
    from scripts.llama_finetune.config import format_system_prompt

    return format_system_prompt()


def build_prompt(tokenizer, system_prompt: str, input_text: str) -> str:
    user_prompt = INSTRUCTION_TEMPLATE.format(input_text=input_text).strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_ollama_request(
    prompt: str,
    model: str,
    api_url: str,
    max_new_tokens: int,
    timeout_seconds: int,
    retries: int,
    backoff_seconds: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
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


def ensure_chunks(case_path: Path, chunk_dir: Path, chunk_tokens: int, timing_path: Path) -> None:
    if chunk_dir.exists() and list(chunk_dir.glob("chunk_*.txt")):
        return

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "chunk.py",
        "--input-file",
        str(case_path),
        "--output-dir",
        str(chunk_dir),
        "--max-tokens",
        str(chunk_tokens),
        "--min-last-chunk-words",
        "20",
        "--use-tokenizer",
    ]
    run_cmd(cmd, LINKKG_DIR, timing_path, {
        "stage": "chunk",
        "case": case_path.stem,
        "entity_type": "all",
        "approach": "shared",
    })


def run_nerre_extraction(
    case_path: Path,
    chunk_dir: Path,
    output_dir: Path,
    model: str,
    tokenizer_name: str,
    max_new_tokens: int,
    api_url: str,
    timing_path: Path,
    timeout_seconds: int,
    retries: int,
    backoff_seconds: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    extraction_csv = output_dir / "extraction.csv"
    if extraction_csv.exists():
        return extraction_csv

    system_prompt = load_system_prompt()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chunk_files = sorted(chunk_dir.glob("chunk_*.txt"))
    rows = []

    stage_start = time.time()
    for idx, chunk_path in enumerate(chunk_files, start=1):
        chunk_text = chunk_path.read_text(encoding="utf-8")
        prompt_str = build_prompt(tokenizer, system_prompt, chunk_text)

        start = time.time()
        output_text = run_ollama_request(
            prompt=prompt_str,
            model=model,
            api_url=api_url,
            max_new_tokens=max_new_tokens,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
        )
        end = time.time()

        rows.append({
            "chunk_number": idx,
            "input_text": chunk_text,
            "output_text": output_text,
        })

        record_stage(timing_path, {
            "stage": "nerre_chunk",
            "case": case_path.stem,
            "entity_type": "all",
            "approach": "nerre",
            "chunk_number": idx,
            "start_time": datetime.fromtimestamp(start).isoformat(),
            "end_time": datetime.fromtimestamp(end).isoformat(),
            "duration_seconds": round(end - start, 2),
            "model": model,
        })

        extraction_csv.parent.mkdir(parents=True, exist_ok=True)
        with extraction_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_number", "input_text", "output_text"])
            writer.writeheader()
            writer.writerows(rows)

    stage_end = time.time()
    record_stage(timing_path, {
        "stage": "nerre_extraction",
        "case": case_path.stem,
        "entity_type": "all",
        "approach": "nerre",
        "start_time": datetime.fromtimestamp(stage_start).isoformat(),
        "end_time": datetime.fromtimestamp(stage_end).isoformat(),
        "duration_seconds": round(stage_end - stage_start, 2),
        "model": model,
        "chunks": len(chunk_files),
    })

    return extraction_csv


def entity_prompt_paths(entity_type: str) -> Dict[str, Path]:
    prefix = ENTITY_CONFIGS[entity_type]
    return {
        "ner": PROMPTS_DIR / f"{prefix}_ner_prompt.txt",
        "coref": PROMPTS_DIR / f"{prefix}_coref_prompt.txt",
        "resolve": PROMPTS_DIR / f"{prefix}_resolve_prompt.txt",
    }


def write_case_manifest(
    output_root: Path,
    case_name: str,
    case_path: Path,
    approaches: List[str],
    chunk_tokens: int,
    nerre_model: str,
    model_70b: str,
    tokenizer_name: str,
) -> None:
    manifest = {
        "case": case_name,
        "input_file": str(case_path),
        "approaches": approaches,
        "chunk_tokens": chunk_tokens,
        "nerre_model": nerre_model,
        "mapping_model": model_70b,
        "resolve_model": model_70b,
        "tokenizer": tokenizer_name,
        "prompts": {},
    }

    for entity_type in ENTITY_CONFIGS.keys():
        paths = entity_prompt_paths(entity_type)
        manifest["prompts"][entity_type] = {
            "ner": str(paths["ner"]),
            "ner_sha256": sha256_file(paths["ner"]),
            "coref": str(paths["coref"]),
            "coref_sha256": sha256_file(paths["coref"]),
            "resolve": str(paths["resolve"]),
            "resolve_sha256": sha256_file(paths["resolve"]),
        }

    manifest_path = output_root / "common" / case_name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_full_70b(
    case_name: str,
    chunk_dir: Path,
    output_root: Path,
    model_name: str,
    timing_path: Path,
    skip_existing: bool,
) -> None:
    approach = "full_70b"
    approach_dir = output_root / approach / case_name
    done_marker = approach_dir / "_DONE"
    if skip_existing and done_marker.exists():
        return

    for entity_type in ENTITY_CONFIGS.keys():
        paths = entity_prompt_paths(entity_type)
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing prompt file: {path}")

        output_dir = approach_dir / entity_type
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "log.txt"

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
            "approach": approach,
            "model": model_name,
        })

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
            "approach": approach,
            "model": model_name,
        })

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
            "approach": approach,
            "model": model_name,
        })

    from generate_kgs import create_consolidated_kg, visualize_kg, save_kg_as_graphml, save_kg_stats

    kg_output_dir = output_root / "kgs" / approach / case_name
    kg_output_dir.mkdir(parents=True, exist_ok=True)
    kg_start = time.time()
    G, entities_by_type = create_consolidated_kg(str(output_root / approach), case_name, str(kg_output_dir))
    if G.number_of_nodes() > 0:
        visualize_kg(G, str(kg_output_dir / f"{case_name}_kg.png"), case_name)
        save_kg_as_graphml(G, str(kg_output_dir / f"{case_name}_kg.graphml"))
        save_kg_stats(G, str(kg_output_dir / "kg_stats.json"), entities_by_type)
    kg_end = time.time()
    record_stage(timing_path, {
        "stage": "kg",
        "case": case_name,
        "entity_type": "all",
        "approach": approach,
        "start_time": datetime.fromtimestamp(kg_start).isoformat(),
        "end_time": datetime.fromtimestamp(kg_end).isoformat(),
        "duration_seconds": round(kg_end - kg_start, 2),
    })

    done_marker.write_text("ok")


def run_full_finetuned(
    case_name: str,
    extraction_csv: Path,
    output_root: Path,
    model_name: str,
    timing_path: Path,
    skip_existing: bool,
) -> None:
    approach = "full_finetuned"
    approach_dir = output_root / approach / case_name
    done_marker = approach_dir / "_DONE"
    if skip_existing and done_marker.exists():
        return

    for entity_type in ENTITY_CONFIGS.keys():
        paths = entity_prompt_paths(entity_type)
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing prompt file: {path}")

        output_dir = approach_dir / entity_type
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "log.txt"

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
            "approach": approach,
        })

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
            "approach": approach,
            "model": model_name,
        })

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
            "approach": approach,
            "model": model_name,
        })

    from generate_kgs import create_consolidated_kg, visualize_kg, save_kg_as_graphml, save_kg_stats

    kg_output_dir = output_root / "kgs" / approach / case_name
    kg_output_dir.mkdir(parents=True, exist_ok=True)
    kg_start = time.time()
    G, entities_by_type = create_consolidated_kg(str(output_root / approach), case_name, str(kg_output_dir))
    if G.number_of_nodes() > 0:
        visualize_kg(G, str(kg_output_dir / f"{case_name}_kg.png"), case_name)
        save_kg_as_graphml(G, str(kg_output_dir / f"{case_name}_kg.graphml"))
        save_kg_stats(G, str(kg_output_dir / "kg_stats.json"), entities_by_type)
    kg_end = time.time()
    record_stage(timing_path, {
        "stage": "kg",
        "case": case_name,
        "entity_type": "all",
        "approach": approach,
        "start_time": datetime.fromtimestamp(kg_start).isoformat(),
        "end_time": datetime.fromtimestamp(kg_end).isoformat(),
        "duration_seconds": round(kg_end - kg_start, 2),
    })

    done_marker.write_text("ok")


def run_shortcut(
    case_name: str,
    extraction_csv: Path,
    output_root: Path,
    model_name: str,
    timing_path: Path,
    skip_existing: bool,
) -> None:
    approach = "shortcut"
    approach_dir = output_root / approach / case_name
    done_marker = approach_dir / "_DONE"
    if skip_existing and done_marker.exists():
        return

    for entity_type in ENTITY_CONFIGS.keys():
        paths = entity_prompt_paths(entity_type)
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing prompt file: {path}")

        output_dir = approach_dir / entity_type
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "log.txt"

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
            "approach": approach,
        })

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
            "approach": approach,
            "model": model_name,
        })

    from generate_kgs_from_csv import create_kg_from_extraction, visualize_kg, save_kg_stats, save_kg_graphml

    kg_output_dir = output_root / "kgs" / approach / case_name
    kg_output_dir.mkdir(parents=True, exist_ok=True)
    kg_start = time.time()
    G, entities_data, alias_stats = create_kg_from_extraction(
        case_name,
        str(extraction_csv),
        str(output_root / approach),
    )
    if G.number_of_nodes() > 0:
        visualize_kg(G, str(kg_output_dir / f"{case_name}_kg.png"), case_name)
        save_kg_stats(G, str(kg_output_dir / "kg_stats.json"), entities_data, alias_stats=alias_stats)
        save_kg_graphml(G, str(kg_output_dir / f"{case_name}_kg.graphml"))
    kg_end = time.time()
    record_stage(timing_path, {
        "stage": "kg",
        "case": case_name,
        "entity_type": "all",
        "approach": approach,
        "start_time": datetime.fromtimestamp(kg_start).isoformat(),
        "end_time": datetime.fromtimestamp(kg_end).isoformat(),
        "duration_seconds": round(kg_end - kg_start, 2),
        "alias_stats": alias_stats,
    })

    done_marker.write_text("ok")


def load_cases(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LinkKG-HS benchmark pipelines")
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--approaches", nargs="+", default=DEFAULT_APPROACHES)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--nerre-model", default=DEFAULT_NERRE_MODEL)
    parser.add_argument("--ner-model-70b", default=DEFAULT_70B_MODEL)
    parser.add_argument("--coref-model", default=DEFAULT_70B_MODEL)
    parser.add_argument("--resolve-model", default=DEFAULT_70B_MODEL)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff", type=int, default=DEFAULT_BACKOFF)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ollama-url", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    cases = load_cases(input_dir)
    if not cases:
        raise RuntimeError(f"No cases found in {input_dir}")

    if args.case_name:
        cases = [p for p in cases if p.stem == args.case_name]
        if not cases:
            raise RuntimeError(f"Case not found: {args.case_name}")

    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(cases):
            raise ValueError(f"case-index out of range (0-{len(cases)-1})")
        cases = [cases[args.case_index]]

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_root = REPO_ROOT / "runs" / "pipeline_benchmark" / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    api_url = args.ollama_url
    if api_url is None:
        host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
        api_url = f"http://{host}/api/generate"

    for case_path in cases:
        case_name = case_path.stem
        timing_path = output_root / "timing" / f"{case_name}.jsonl"

        common_chunk_dir = output_root / "common" / case_name / "chunk_outputs"
        ensure_chunks(case_path, common_chunk_dir, args.chunk_tokens, timing_path)

        write_case_manifest(
            output_root=output_root,
            case_name=case_name,
            case_path=case_path,
            approaches=args.approaches,
            chunk_tokens=args.chunk_tokens,
            nerre_model=args.nerre_model,
            model_70b=args.coref_model,
            tokenizer_name=args.tokenizer_name,
        )

        if "full_finetuned" in args.approaches or "shortcut" in args.approaches:
            extraction_dir = output_root / "nerre_extractions" / case_name
            extraction_csv = run_nerre_extraction(
                case_path=case_path,
                chunk_dir=common_chunk_dir,
                output_dir=extraction_dir,
                model=args.nerre_model,
                tokenizer_name=args.tokenizer_name,
                max_new_tokens=args.max_new_tokens,
                api_url=api_url,
                timing_path=timing_path,
                timeout_seconds=args.timeout,
                retries=args.retries,
                backoff_seconds=args.backoff,
            )
        else:
            extraction_csv = None

        if "full_70b" in args.approaches:
            run_full_70b(
                case_name=case_name,
                chunk_dir=common_chunk_dir,
                output_root=output_root,
                model_name=args.ner_model_70b,
                timing_path=timing_path,
                skip_existing=args.skip_existing,
            )

        if "full_finetuned" in args.approaches:
            if extraction_csv is None:
                raise RuntimeError("Missing extraction.csv for full_finetuned approach")
            run_full_finetuned(
                case_name=case_name,
                extraction_csv=extraction_csv,
                output_root=output_root,
                model_name=args.coref_model,
                timing_path=timing_path,
                skip_existing=args.skip_existing,
            )

        if "shortcut" in args.approaches:
            if extraction_csv is None:
                raise RuntimeError("Missing extraction.csv for shortcut approach")
            run_shortcut(
                case_name=case_name,
                extraction_csv=extraction_csv,
                output_root=output_root,
                model_name=args.coref_model,
                timing_path=timing_path,
                skip_existing=args.skip_existing,
            )


if __name__ == "__main__":
    main()
