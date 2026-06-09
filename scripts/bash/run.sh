#!/bin/bash
#SBATCH --job-name=linkkg-repair-batch
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=72:00:00

set -euo pipefail

APPROACHES="${APPROACHES:-full_70b full_finetuned}"
MODEL_70B="${MODEL_70B:-llama3.1:70b}"
FORCE="0"
SKIP_GRAPHRAG="${SKIP_GRAPHRAG:-0}"
OLLAMA_SKIP_PULL="${OLLAMA_SKIP_PULL:-0}"
USE_FAILED_LIST="${USE_FAILED_LIST:-0}"
MAX_RETRIES="${MAX_RETRIES:-1}"
export GRAPHRAG_API_KEY="NONE"
export OLLAMA_CONTEXT_LENGTH=16384

ROOT="/scratch/efeldma5/uniner_project/link-kg"
FAILED_LIST="$ROOT/tmp_failed_logs.txt"

SOURCE_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "$(basename "$SOURCE_ROOT")" == "setup" ]]; then
    SOURCE_ROOT="$(dirname "$SOURCE_ROOT")"
fi

SKIP_RSYNC="0"

if [[ "$SOURCE_ROOT" == "$ROOT" ]]; then
    echo "INFO: SOURCE_ROOT == ROOT ($ROOT); skipping rsync."
    SKIP_RSYNC="1"
elif [[ "$SOURCE_ROOT" == "$ROOT/"* ]]; then
    echo "ERROR: SOURCE_ROOT is inside ROOT ($ROOT)."
    echo "Submit from a different path (e.g., your home repo) or set SOURCE_ROOT explicitly."
    exit 1
elif [[ "$ROOT" == "$SOURCE_ROOT/"* ]]; then
    echo "ERROR: ROOT ($ROOT) is inside SOURCE_ROOT ($SOURCE_ROOT)."
    echo "Submitting from the parent directory will create recursive copies."
    exit 1
fi

if [[ "$SKIP_RSYNC" != "1" ]]; then
    mkdir -p "$ROOT"
    rsync -a --delete \
        --exclude 'runs/' \
        --exclude 'slurm_logs/' \
        --exclude '.git/' \
        "$SOURCE_ROOT"/ "$ROOT"/
fi

if [[ "$USE_FAILED_LIST" == "1" && ! -f "$FAILED_LIST" ]]; then
    echo "Missing failed list at $FAILED_LIST (USE_FAILED_LIST=1)."
    exit 1
fi

module load hosts/hopper gnu/12.3.0
module load cuda/12.6.3
module load ollama/0.20.3
module load python/3.12.1-33

export TIKTOKEN_CACHE_DIR=/scratch/efeldma5/uniner_project/tiktoken_cache
export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/runs/ollama_logs"
OLLAMA_LOG="$LOG_DIR/ollama_repair_batch_${SLURM_JOB_ID}.log"
mkdir -p "$OLLAMA_MODELS"
mkdir -p "$LOG_DIR"

ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    if ps -p $OLLAMA_PID > /dev/null 2>&1; then
        kill $OLLAMA_PID
        wait $OLLAMA_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for Ollama to start..."
until curl -s --fail "http://127.0.0.1:11434/api/tags" > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is up!"

if [[ "$OLLAMA_SKIP_PULL" != "1" ]]; then
    ollama pull "$MODEL_70B"
    ollama pull nomic-embed-text
fi
source /scratch/efeldma5/uniner_project/venv/bin/activate
nvidia-smi
cd /scratch/efeldma5/uniner_project/link-kg/LinkKG-HS/linkkg
mkdir -p /scratch/efeldma5/uniner_project/link-kg/slurm_logs

RUNS_ROOT="$ROOT/runs/pipeline_benchmark"

declare -a CASE_ROWS=()

if [[ "$USE_FAILED_LIST" == "1" && -f "$FAILED_LIST" ]]; then
    while read -r run_id case_name; do
        [[ -z "$run_id" || -z "$case_name" ]] && continue
        CASE_ROWS+=("$run_id:$case_name")
    done < <(grep -E '^[0-9]+/' "$FAILED_LIST" | awk '{split($1,a,"/"); print a[1],a[2]}' | sort -u)
else
    while IFS= read -r -d '' run_dir; do
        run_id="$(basename "$run_dir")"
        common_dir="$run_dir/common"
        [[ -d "$common_dir" ]] || continue
        case_path="$(find "$common_dir" -mindepth 1 -maxdepth 1 -type d -print -quit)"
        [[ -n "$case_path" ]] || continue
        case_name="$(basename "$case_path")"
        CASE_ROWS+=("$run_id:$case_name")
    done < <(find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

for row in "${CASE_ROWS[@]}"; do
    run_id="${row%%:*}"
    case_name="${row#*:}"
    [[ -z "$run_id" || -z "$case_name" ]] && continue

    run_root="$RUNS_ROOT/$run_id"
    [[ -d "$run_root" ]] || continue

    missing=()
    for approach in $APPROACHES; do
        kg_dir="$run_root/kgs/$approach/$case_name"
        if [[ ! -f "$kg_dir/${case_name}_kg.graphml" && ! -f "$kg_dir/kg_stats.json" ]]; then
            missing+=("$approach")
        fi
    done

    if [[ "${#missing[@]}" -eq 0 && "$FORCE" != "1" ]]; then
        echo "=== Skip $run_id / $case_name (all KGs present) ==="
        continue
    fi

    echo "=== Repairing $run_id / $case_name (missing: ${missing[*]:-all}) ==="
    ARGS=(--run-id "$run_id" --case "$case_name" --model-70b "$MODEL_70B")
    if [[ "${#missing[@]}" -gt 0 ]]; then
        ARGS+=(--approach "${missing[@]}")
    else
        ARGS+=(--approach $APPROACHES)
    fi

    if [[ "$FORCE" == "1" ]]; then
        ARGS+=(--force)
    fi
    if [[ "$SKIP_GRAPHRAG" == "1" ]]; then
        ARGS+=(--skip-graphrag)
    fi

    attempt=0
    while true; do
        if python repair_pipeline_benchmark.py "${ARGS[@]}"; then
            echo "=== Done $run_id / $case_name ==="
            break
        fi

        if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
            echo "=== Failed $run_id / $case_name after $((attempt + 1)) attempts ==="
            break
        fi

        attempt=$((attempt + 1))
        echo "=== Retry $attempt for $run_id / $case_name ==="
    done
done
