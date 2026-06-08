import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]

OLLAMA_MODEL = "2028efeldman/llama-finetuned:latest"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
INPUT_DIR = REPO_ROOT / "datasets" / "cleanedinput"
OUTPUT_ROOT = REPO_ROOT / "runs" / "eval_llama_finetuned"
CHUNK_SIZE = 300
MAX_NEW_TOKENS = 4000
START_CASE_NUMBER = 1

REQUEST_TIMEOUT_SECONDS = 600
REQUEST_RETRIES = 5
REQUEST_RETRY_BACKOFF_SECONDS = 2


def chunk_text_by_tokens(text: str, chunk_size: int) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []

    chunks: List[str] = []
    for start in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[start : start + chunk_size]
        chunk_text = " ".join(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def run_inference_ollama(input_text: str, max_new_tokens: int) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": input_text,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_new_tokens,
        },
    }

    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_exc: Exception = RuntimeError("Unknown Ollama request error")
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode("utf-8"))
            output = body.get("response", "")
            if not isinstance(output, str):
                raise RuntimeError(f"Unexpected Ollama response format: {body}")
            return output.strip()
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as exc:
            last_exc = exc
            if attempt == REQUEST_RETRIES:
                break
            wait_seconds = REQUEST_RETRY_BACKOFF_SECONDS * attempt
            print(
                f"    ! Ollama request failed (attempt {attempt}/{REQUEST_RETRIES}): {exc}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to call Ollama API after {REQUEST_RETRIES} attempts: {last_exc}"
    )


def iter_case_files(input_dir: Path) -> List[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def write_case_csv(output_csv_path: Path, rows: List[Dict]) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["chunk_number", "input_text", "output_text"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    input_dir = INPUT_DIR
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = OUTPUT_ROOT / run_timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)

    case_files = iter_case_files(input_dir)
    if not case_files:
        raise RuntimeError(f"No .txt files found in input directory: {input_dir}")
    if START_CASE_NUMBER < 1 or START_CASE_NUMBER > len(case_files):
        raise ValueError(
            f"START_CASE_NUMBER must be in [1, {len(case_files)}], got {START_CASE_NUMBER}"
        )

    selected_case_files = case_files[START_CASE_NUMBER - 1 :]

    print(f"Input directory: {input_dir}")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print(f"Ollama API: {OLLAMA_API_URL}")
    print(f"Run output: {run_output_dir}")
    print(f"Cases found: {len(case_files)}")
    print(f"Starting from case: {START_CASE_NUMBER}")

    for offset, case_path in enumerate(selected_case_files, start=0):
        case_idx = START_CASE_NUMBER + offset
        case_name = case_path.stem
        raw_text = case_path.read_text(encoding="utf-8").strip()
        chunks = chunk_text_by_tokens(raw_text, CHUNK_SIZE)

        case_csv_path = run_output_dir / case_name / "extraction.csv"
        case_rows: List[Dict] = []

        print(f"[{case_idx}/{len(case_files)}] {case_name}: {len(chunks)} chunks")

        for idx, chunk_text in enumerate(chunks, start=1):
            print(f"  - chunk {idx}/{len(chunks)}")
            output_text = run_inference_ollama(chunk_text, MAX_NEW_TOKENS)
            case_rows.append(
                {
                    "chunk_number": idx,
                    "input_text": chunk_text,
                    "output_text": output_text,
                }
            )
            write_case_csv(case_csv_path, case_rows)

        print(f"[{case_name}] chunks={len(case_rows)} -> {case_csv_path}")


if __name__ == "__main__":
    main()
