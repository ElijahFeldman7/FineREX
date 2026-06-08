import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import urllib.error
import urllib.request
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
LINKKG_DIR = REPO_ROOT / "LinkKG-HS" / "linkkg"
RUNS_ROOT = REPO_ROOT / "runs" / "pipeline_benchmark"
PROMPTS_DIR = LINKKG_DIR / "prompts"
ALL_ENTITIES_PROMPT = PROMPTS_DIR / "all_entities_resolve_prompt.txt"

ENTITY_TYPES = [
    "person",
    "location",
    "organization",
    "routes",
    "means_of_transportation",
    "means_of_communication",
    "smuggled_items",
]

DEFAULT_FINETUNED_MODEL = "2028efeldman/llama-finetuned:latest"
DEFAULT_70B_MODEL = "llama3.1:70b"
DEFAULT_TOKENIZER_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 4000
DEFAULT_CHUNK_TOKENS = 300
DEFAULT_TIMEOUT = 600
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 2

INSTRUCTION_TEMPLATE = """Input_text: \n{input_text}\nOutput:\n"""


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

    raise RuntimeError(f"Failed to call Ollama API after {retries} attempts: {last_exc}")


def merge_description(existing: str, incoming: str) -> str:
    existing = (existing or "").strip()
    incoming = (incoming or "").strip()

    if not existing:
        return incoming
    if not incoming:
        return existing
    if incoming in existing:
        return existing
    if existing in incoming:
        return incoming
    return f"{existing} | {incoming}"


def load_final_memory(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {"RESOLVED_ENTITIES": {}, "AUXILIARY_DESCRIPTIONS": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "RESOLVED_ENTITIES": data.get("RESOLVED_ENTITIES", {}) or {},
        "AUXILIARY_DESCRIPTIONS": data.get("AUXILIARY_DESCRIPTIONS", {}) or {},
    }


def merge_memories(target: Dict[str, Dict], incoming: Dict[str, Dict]) -> None:
    resolved = target["RESOLVED_ENTITIES"]
    aux_desc = target["AUXILIARY_DESCRIPTIONS"]

    for alias, canonical in (incoming.get("RESOLVED_ENTITIES") or {}).items():
        if canonical is None:
            continue
        existing = resolved.get(alias)
        if existing is None:
            resolved[alias] = canonical
        elif existing != canonical:
            resolved[alias] = canonical if len(str(canonical)) > len(str(existing)) else existing

    for key, desc in (incoming.get("AUXILIARY_DESCRIPTIONS") or {}).items():
        if key in aux_desc:
            aux_desc[key] = merge_description(aux_desc[key], desc)
        else:
            aux_desc[key] = desc


def build_all_entities_memory(
    run_root: Path,
    approach: str,
    case_name: str,
    timing_path: Path,
    force: bool,
) -> Path:
    all_entities_dir = run_root / approach / case_name / "all_entities"
    memory_path = all_entities_dir / "final_memory.json"
    if memory_path.exists() and not force:
        return memory_path

    start = time.time()
    combined = {"RESOLVED_ENTITIES": {}, "AUXILIARY_DESCRIPTIONS": {}}
    for entity_type in ENTITY_TYPES:
        mem_path = run_root / approach / case_name / entity_type / "final_memory.json"
        incoming = load_final_memory(mem_path)
        merge_memories(combined, incoming)

    all_entities_dir.mkdir(parents=True, exist_ok=True)
    with memory_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    end = time.time()
    record_stage(timing_path, {
        "stage": "all_entities_memory",
        "case": case_name,
        "entity_type": "all",
        "approach": approach,
        "start_time": datetime.fromtimestamp(start).isoformat(),
        "end_time": datetime.fromtimestamp(end).isoformat(),
        "duration_seconds": round(end - start, 2),
    })

    return memory_path


def ensure_chunk_outputs_link(output_dir: Path, source_dir: Path) -> None:
    chunk_dir = output_dir / "chunk_outputs"
    if chunk_dir.exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source_dir, chunk_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(source_dir, chunk_dir)


def ensure_all_entities_resolved(
    run_root: Path,
    approach: str,
    case_name: str,
    timing_path: Path,
    resolve_model: str,
    force: bool,
) -> Path:
    all_entities_dir = run_root / approach / case_name / "all_entities"
    resolved_path = all_entities_dir / f"all_entities_resolved_{case_name}.txt"
    if resolved_path.exists() and not force:
        return resolved_path

    if not ALL_ENTITIES_PROMPT.exists():
        raise FileNotFoundError(f"Missing resolve prompt: {ALL_ENTITIES_PROMPT}")

    chunks_dir = run_root / "common" / case_name / "chunk_outputs"
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Missing chunks dir: {chunks_dir}")

    ensure_chunk_outputs_link(all_entities_dir, chunks_dir)

    memory_path = build_all_entities_memory(
        run_root=run_root,
        approach=approach,
        case_name=case_name,
        timing_path=timing_path,
        force=force,
    )

    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--input-file-name",
        case_name,
        "--entity-type",
        "all_entities",
        "--output-dir",
        str(all_entities_dir),
        "--final-memory-dir",
        str(all_entities_dir),
        "--resolve-prompt-file",
        str(ALL_ENTITIES_PROMPT),
        "--resolve-model-name",
        resolve_model,
        "--resolve-num-retries",
        "1",
        "--run-stages",
        "resolve",
    ]
    run_cmd(cmd, LINKKG_DIR, timing_path, {
        "stage": "all_entities_resolve",
        "case": case_name,
        "entity_type": "all",
        "approach": approach,
        "model": resolve_model,
    })

    return resolved_path


def ensure_chunks(
    resolved_path: Path,
    chunk_dir: Path,
    chunk_tokens: int,
    timing_path: Path,
    case_name: str,
    approach_tag: str,
    force: bool,
) -> None:
    if not force and chunk_dir.exists() and list(chunk_dir.glob("chunk_*.txt")):
        return

    chunk_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "chunk.py",
        "--input-file",
        str(resolved_path),
        "--output-dir",
        str(chunk_dir),
        "--max-tokens",
        str(chunk_tokens),
        "--min-last-chunk-words",
        "20",
        "--use-tokenizer",
    ]
    run_cmd(cmd, LINKKG_DIR, timing_path, {
        "stage": "resolved_chunk",
        "case": case_name,
        "entity_type": "all",
        "approach": approach_tag,
    })


def run_nerre_extraction(
    case_name: str,
    approach_tag: str,
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
    force: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    extraction_csv = output_dir / "extraction.csv"
    if extraction_csv.exists() and not force:
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
            "stage": "resolved_nerre_chunk",
            "case": case_name,
            "entity_type": "all",
            "approach": approach_tag,
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
        "stage": "resolved_nerre_extraction",
        "case": case_name,
        "entity_type": "all",
        "approach": approach_tag,
        "start_time": datetime.fromtimestamp(stage_start).isoformat(),
        "end_time": datetime.fromtimestamp(stage_end).isoformat(),
        "duration_seconds": round(stage_end - stage_start, 2),
        "model": model,
        "chunks": len(chunk_files),
    })

    return extraction_csv


def run_kg_from_extraction(
    case_name: str,
    approach_tag: str,
    kg_approach: str,
    output_tag: str,
    extraction_csv: Path,
    processed_kg_dir: Path,
    run_root: Path,
    timing_path: Path,
    force: bool,
) -> None:
    from generate_kgs_from_csv import (
        create_kg_from_extraction,
        visualize_kg,
        save_kg_stats,
        save_kg_graphml,
    )

    kg_output_dir = run_root / "kgs" / kg_approach / case_name
    graphml_path = kg_output_dir / f"{case_name}_{output_tag}.graphml"
    if graphml_path.exists() and not force:
        return

    kg_output_dir.mkdir(parents=True, exist_ok=True)

    kg_start = time.time()
    G, entities_data, alias_stats = create_kg_from_extraction(
        case_name,
        str(extraction_csv),
        str(processed_kg_dir),
    )
    if G.number_of_nodes() > 0:
        visualize_kg(G, str(kg_output_dir / f"{case_name}_{output_tag}.png"), case_name)
        save_kg_stats(
            G,
            str(kg_output_dir / f"kg_stats_{output_tag}.json"),
            entities_data,
            alias_stats=alias_stats,
        )
        save_kg_graphml(G, str(graphml_path))
    kg_end = time.time()

    record_stage(timing_path, {
        "stage": "resolved_nerre_kg",
        "case": case_name,
        "entity_type": "all",
        "approach": approach_tag,
        "start_time": datetime.fromtimestamp(kg_start).isoformat(),
        "end_time": datetime.fromtimestamp(kg_end).isoformat(),
        "duration_seconds": round(kg_end - kg_start, 2),
        "alias_stats": alias_stats,
    })


def iter_cases(runs_root: Path) -> Iterable[Path]:
    return sorted([p for p in runs_root.iterdir() if p.is_dir()])


def resolved_text_path(run_root: Path, approach: str, case_name: str, source: str) -> Path:
    if source == "smuggled_items":
        return run_root / approach / case_name / "smuggled_items" / f"smuggled_items_resolved_{case_name}.txt"
    if source == "all_entities":
        return run_root / approach / case_name / "all_entities" / f"all_entities_resolved_{case_name}.txt"
    raise ValueError(f"Unknown resolved source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NER-RE over resolved smuggled_items text")
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--approaches", nargs="+", default=["full_70b", "full_finetuned"])
    parser.add_argument("--nerre-model-finetuned", default=DEFAULT_FINETUNED_MODEL)
    parser.add_argument("--nerre-model-70b", default=DEFAULT_70B_MODEL)
    parser.add_argument("--final-model", default=None)
    parser.add_argument("--resolve-model", default=DEFAULT_70B_MODEL)
    parser.add_argument("--resolved-source", default="all_entities", choices=["all_entities", "smuggled_items"])
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--backoff", type=int, default=DEFAULT_BACKOFF)
    parser.add_argument("--processed-kg-dir", default=str(REPO_ROOT / "datasets" / "processed_kg"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ollama-url", default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    processed_kg_dir = Path(args.processed_kg_dir)

    api_url = args.ollama_url
    if api_url is None:
        host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
        api_url = f"http://{host}/api/generate"

    for run_root in iter_cases(runs_root):
        case_name = run_root.name
        timing_path = run_root / "timing" / f"{case_name}.jsonl"

        for approach in args.approaches:
            if args.resolved_source == "all_entities":
                resolved_path = ensure_all_entities_resolved(
                    run_root=run_root,
                    approach=approach,
                    case_name=case_name,
                    timing_path=timing_path,
                    resolve_model=args.resolve_model,
                    force=args.force,
                )
            else:
                resolved_path = resolved_text_path(run_root, approach, case_name, args.resolved_source)
                if not resolved_path.exists():
                    print(f"Skip {case_name}/{approach}: missing {resolved_path}")
                    continue

            if args.final_model:
                model = args.final_model
            elif approach == "full_70b":
                model = args.nerre_model_70b
            elif approach == "full_finetuned":
                model = args.nerre_model_finetuned
            else:
                print(f"Skip {case_name}: unknown approach {approach}")
                continue

            approach_tag = f"resolved_nerre_{approach}"
            output_tag = f"resolved_nerre_{approach}"
            output_dir = run_root / approach_tag / case_name
            chunk_dir = output_dir / "chunk_outputs"
            extraction_csv = output_dir / "extraction.csv"
            graphml_path = run_root / "kgs" / approach / case_name / f"{case_name}_{output_tag}.graphml"

            if args.skip_existing and extraction_csv.exists() and graphml_path.exists():
                print(f"Skip {case_name}/{approach_tag}: outputs already present")
                continue

            ensure_chunks(
                resolved_path=resolved_path,
                chunk_dir=chunk_dir,
                chunk_tokens=args.chunk_tokens,
                timing_path=timing_path,
                case_name=case_name,
                approach_tag=approach_tag,
                force=args.force,
            )

            extraction_csv = run_nerre_extraction(
                case_name=case_name,
                approach_tag=approach_tag,
                chunk_dir=chunk_dir,
                output_dir=output_dir,
                model=model,
                tokenizer_name=args.tokenizer_name,
                max_new_tokens=args.max_new_tokens,
                api_url=api_url,
                timing_path=timing_path,
                timeout_seconds=args.timeout,
                retries=args.retries,
                backoff_seconds=args.backoff,
                force=args.force,
            )

            run_kg_from_extraction(
                case_name=case_name,
                approach_tag=approach_tag,
                kg_approach=approach,
                output_tag=output_tag,
                extraction_csv=extraction_csv,
                processed_kg_dir=processed_kg_dir,
                run_root=run_root,
                timing_path=timing_path,
                force=args.force,
            )


if __name__ == "__main__":
    main()
