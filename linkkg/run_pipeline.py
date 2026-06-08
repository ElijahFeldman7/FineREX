import argparse
import subprocess
import os
import sys
import time
from datetime import datetime

# STEP 1: Logging utility
def log(message, log_file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(full_msg + '\n')

# STEP 1.5: Prep Stage
def run_prep_stage(args, output_dir, log_file):
    log("Starting Prep Step from CSV...", log_file)
    start = time.time()

    prep_cmd = [
        sys.executable, "prep_from_csv.py",
        "--csv-file", args.input_csv,
        "--output-dir", output_dir
    ]
    if args.entity_type:
        prep_cmd.extend(["--entity-type", args.entity_type])

    result = subprocess.run(prep_cmd)
    end = time.time()

    if result.returncode == 0:
        log(f"Prep completed in {end - start:.2f} seconds.", log_file)
    else:
        log("Prep failed.", log_file)
        exit(1)

# STEP 2: Chunking Stage
def run_chunk_stage(args, input_file_name, output_dir, log_file):
    log("Starting Chunking Step...", log_file)
    start = time.time()

    chunk_output_dir = os.path.join(output_dir, "chunk_outputs")
    os.makedirs(chunk_output_dir, exist_ok=True)

    chunk_cmd = [
        sys.executable, "chunk.py",
        "--input-file", args.input_file,
        "--output-dir", chunk_output_dir,
        "--max-tokens", str(args.max_tokens),
        "--min-last-chunk-words", str(args.min_last_chunk_words)
    ]
    if args.use_tokenizer:
        chunk_cmd.append("--use-tokenizer")

    result = subprocess.run(chunk_cmd)
    end = time.time()

    if result.returncode == 0:
        log(f"Chunking completed in {end - start:.2f} seconds.", log_file)
    else:
        log("Chunking failed.", log_file)
        exit(1)

    return chunk_output_dir

# STEP 3: NER Stage
def run_ner_stage(args, chunk_output_dir, output_dir, log_file):
    log("Starting NER Step...", log_file)
    start = time.time()

    ner_output_dir = os.path.join(output_dir, "ner_outputs")
    os.makedirs(ner_output_dir, exist_ok=True)

    ner_cmd = [
        sys.executable, "ner.py",
        "--chunks-dir", chunk_output_dir,
        "--prompt-file", args.ner_prompt_file,
        "--output-dir", ner_output_dir,
        "--log-file", log_file,
        "--model-name", args.ner_model_name,
        "--max-retries", str(args.ner_max_retries)
    ]

    result = subprocess.run(ner_cmd)
    end = time.time()

    if result.returncode == 0:
        log(f"NER completed in {end - start:.2f} seconds.", log_file)
    else:
        log("NER failed.", log_file)
        exit(1)

    return ner_output_dir

# STEP 4: Coreference Resolution
def run_coref_stage(args, chunk_output_dir, ner_output_dir, output_dir, input_file_name, log_file):
    log("Starting Coreference Resolution Step...", log_file)
    start = time.time()

    coref_cmd = [
        sys.executable, "loopcoref.py",
        "--chunks-dir", chunk_output_dir,
        "--ner-dir", ner_output_dir,
        "--prompt-file", args.coref_prompt_file,
        "--base-output-folder", output_dir,
        "--input-file-name", input_file_name,
        "--model", args.coref_model_name,
        "--verify-passes", str(args.coref_verify_passes),
        "--log-file", log_file,
        "--max-retries", str(args.coref_max_retries)
    ]

    if args.coref_verify_prompt_file:
        coref_cmd.extend(["--verify-prompt-file", args.coref_verify_prompt_file])

    result = subprocess.run(coref_cmd)
    end = time.time()

    if result.returncode == 0:
        log(f"Coreference Resolution completed in {end - start:.2f} seconds.", log_file)
    else:
        log("Coreference Resolution failed.", log_file)
        exit(1)

# STEP 5: Final Resolution
def run_resolve_stage(args, chunk_output_dir, output_dir, input_file_name, log_file):
    log("Starting Final Coref Resolution Step...", log_file)
    start = time.time()

    resolved_output_dir = os.path.join(output_dir, "resolved_outputs")
    os.makedirs(resolved_output_dir, exist_ok=True)

    # Use final_memory_dir if provided, otherwise fall back to output_dir
    if args.final_memory_dir:
        final_memory_path = os.path.join(args.final_memory_dir, "final_memory.json")
    else:
        final_memory_path = os.path.join(output_dir, "final_memory.json")

    resolve_cmd = [
        sys.executable, "resolve_coref.py",
        "--chunks-dir", chunk_output_dir,
        "--final-memory", final_memory_path,
        "--prompt-file", args.resolve_prompt_file,
        "--base-output-dir", output_dir,
        "--input-file-name", input_file_name,
        "--model-name", args.resolve_model_name,
        "--num-retries", str(args.resolve_num_retries),
        "--num-ctx", "8192",
        "--request-timeout", "600",
        "--log-file", log_file,
        "--entity-type", str(args.entity_type)
    ]

    result = subprocess.run(resolve_cmd)
    end = time.time()

    if result.returncode == 0:
        log(f"Final Resolution completed in {end - start:.2f} seconds.", log_file)
    else:
        log("Final Resolution failed.", log_file)
        exit(1)

# STEP 6: Main Pipeline Controller
def main():
    cumulative_time = 0.0
    parser = argparse.ArgumentParser(description="Run full pipeline on legal text.")
    parser.add_argument("--input-file-name", required=True, help="Name to use for output folder (do not include extension).")
    parser.add_argument("--entity-type", required=True, help="Entity type to process (e.g., person, location, org, etc.)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Use this exact output directory for pipeline inputs/outputs (must contain "
            "chunk_outputs and ner_outputs when skipping chunk/ner). If omitted, uses "
            "output/<input-file-name>/<entity-type>."
        ),
    )
    parser.add_argument(
        "--final-memory-dir",
        default=None,
        help="Directory containing final_memory.json for resolve stage (e.g., datasets/processed_kg/case_name)"
    )

    parser.add_argument("--input-file")
    parser.add_argument("--input-csv", help="Path to input CSV file for 'prep' stage")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--min-last-chunk-words", type=int, default=20)
    parser.add_argument("--use-tokenizer", action="store_true")

    # NER args
    parser.add_argument("--ner-prompt-file")
    parser.add_argument("--ner-model-name")
    parser.add_argument("--ner-max-retries", type=int, default=2)

    # Coref args
    parser.add_argument("--coref-prompt-file")
    parser.add_argument("--coref-verify-prompt-file")
    parser.add_argument("--coref-model-name")
    parser.add_argument("--coref-verify-passes", type=int, default=0)
    parser.add_argument("--coref-max-retries", type=int, default=3)

    # Final Resolution args
    parser.add_argument("--resolve-prompt-file")
    parser.add_argument("--resolve-model-name")
    parser.add_argument("--resolve-num-retries", type=int, default=1, help="Number of times to reprocess each chunk to improve resolution")

    # Stage control
    parser.add_argument("--run-stages", nargs="+", choices=["prep", "chunk", "ner", "coref", "resolve"], required=True)

    args = parser.parse_args()

    #input_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    input_file_name = args.input_file_name
    output_dir = args.output_dir if args.output_dir else os.path.join("output", input_file_name , args.entity_type)
    log_file = os.path.join(output_dir, "log.txt")
    os.makedirs(output_dir, exist_ok=True)

    # Validate only what is needed for requested stages.
    if "prep" in args.run_stages and not args.input_csv:
        raise ValueError("--input-csv is required when running the 'prep' stage.")
    if "chunk" in args.run_stages and not args.input_file:
        raise ValueError("--input-file is required when running the 'chunk' stage.")
    if "ner" in args.run_stages and (not args.ner_prompt_file or not args.ner_model_name):
        raise ValueError("--ner-prompt-file and --ner-model-name are required when running the 'ner' stage.")
    if "coref" in args.run_stages and (not args.coref_prompt_file or not args.coref_model_name):
        raise ValueError("--coref-prompt-file and --coref-model-name are required when running the 'coref' stage.")
    if "resolve" in args.run_stages and (not args.resolve_prompt_file or not args.resolve_model_name):
        raise ValueError("--resolve-prompt-file and --resolve-model-name are required when running the 'resolve' stage.")

    # Prep Stage
    if "prep" in args.run_stages:
        stage_start = time.time()
        run_prep_stage(args, output_dir, log_file)
        stage_end = time.time()
        stage_duration = stage_end - stage_start
        cumulative_time += stage_duration
        log(f"Cumulative time after Prep: {cumulative_time:.2f} seconds.", log_file)

    # Chunk Stage
    if "chunk" in args.run_stages:
        stage_start = time.time()
        chunk_output_dir = run_chunk_stage(args, input_file_name, output_dir, log_file)
        stage_end = time.time()
        stage_duration = stage_end - stage_start
        cumulative_time += stage_duration
        log(f"Cumulative time after Chunking: {cumulative_time:.2f} seconds.", log_file)
    else:
        chunk_output_dir = os.path.join(output_dir, "chunk_outputs")
        if "prep" not in args.run_stages and not os.path.exists(chunk_output_dir):
            raise FileNotFoundError(f"Chunk output not found at {chunk_output_dir}. Run with 'chunk' or 'prep' stage.")

    # NER Stage
    if "ner" in args.run_stages:
        stage_start = time.time()
        ner_output_dir = run_ner_stage(args, chunk_output_dir, output_dir, log_file)
        stage_end = time.time()
        stage_duration = stage_end - stage_start
        cumulative_time += stage_duration
        log(f"Cumulative time after NER: {cumulative_time:.2f} seconds.", log_file)
    else:
        ner_output_dir = os.path.join(output_dir, "ner_outputs")
        if "coref" in args.run_stages and "prep" not in args.run_stages and not os.path.exists(ner_output_dir):
            raise FileNotFoundError(f"NER output not found at {ner_output_dir}. Run with 'ner' or 'prep' stage.")

    # Coref Stage
    if "coref" in args.run_stages:
        stage_start = time.time()
        run_coref_stage(args, chunk_output_dir, ner_output_dir, output_dir, input_file_name, log_file)
        stage_end = time.time()
        stage_duration = stage_end - stage_start
        cumulative_time += stage_duration
        log(f"Cumulative time after Coref: {cumulative_time:.2f} seconds.", log_file)
    else:
        coref_output_dir = os.path.join(output_dir, "coref_outputs")
        if args.final_memory_dir:
            final_memory_path = os.path.join(args.final_memory_dir, "final_memory.json")
        else:
            final_memory_path = os.path.join(output_dir, "final_memory.json")
        if "resolve" in args.run_stages and not os.path.exists(final_memory_path):
            raise FileNotFoundError(f"Final memory not found at {final_memory_path}. Run with 'coref' stage.")

    # Final Resolution Stage
    if "resolve" in args.run_stages:
        stage_start = time.time()
        run_resolve_stage(args, chunk_output_dir, output_dir, input_file_name, log_file)
        stage_end = time.time()
        stage_duration = stage_end - stage_start
        cumulative_time += stage_duration
        log(f"Cumulative time after Final Resolution: {cumulative_time:.2f} seconds.", log_file)
    else:
        resolved_dir = os.path.join(output_dir, "resolved_outputs")
        if not os.path.exists(resolved_dir):
             pass # Not strictly required if not requested

    log(f"Total pipeline time: {cumulative_time:.2f} seconds.", log_file)
if __name__ == "__main__":
    main()
