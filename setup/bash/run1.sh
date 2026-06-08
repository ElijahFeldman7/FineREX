#!/bin/bash
#SBATCH --job-name=linkkg-shortcut-rerun
#SBATCH --output=/scratch/efeldma5/uniner_project/link-kg/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/efeldma5/uniner_project/link-kg/slurm_logs/%x-%j.err
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=72:00:00

set -euo pipefail

CASE_NAME="${CASE_NAME:-16USVsMacmilon}"
APPROACHES="${APPROACHES:-shortcut}"
NERRE_MODEL="${NERRE_MODEL:-2028efeldman/llama-finetuned:latest}"
COREF_MODEL="${COREF_MODEL:-llama3.1:70b}"
CHUNK_TOKENS="${CHUNK_TOKENS:-300}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
OLLAMA_SKIP_PULL="${OLLAMA_SKIP_PULL:-0}"
export GRAPHRAG_API_KEY="NONE"
export OLLAMA_CONTEXT_LENGTH=16384

ROOT="/scratch/efeldma5/uniner_project/link-kg"
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
        --exclude 'slurm_logs/' \
        --exclude '.git/' \
        "$SOURCE_ROOT"/ "$ROOT"/
fi

module load hosts/hopper gnu/12.3.0
module load cuda/12.6.3
module load ollama/0.20.3
module load python/3.12.1-33

export TIKTOKEN_CACHE_DIR=/scratch/efeldma5/uniner_project/tiktoken_cache
export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
LOG_DIR="$ROOT/runs/ollama_logs"
OLLAMA_LOG="$LOG_DIR/ollama_resolved_nerre_${SLURM_JOB_ID}.log"
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
    ollama pull "$NERRE_MODEL"
    ollama pull "$COREF_MODEL"
    ollama pull nomic-embed-text
fi

source /scratch/efeldma5/uniner_project/venv/bin/activate
nvidia-smi
cd "$ROOT/LinkKG-HS/linkkg"
mkdir -p "$ROOT/slurm_logs"

OUTPUT_ROOT="$ROOT/runs/pipeline_benchmark/$CASE_NAME"

ARGS=(
    --case-name "$CASE_NAME"
    --output-root "$OUTPUT_ROOT"
    --approaches $APPROACHES
    --chunk-tokens "$CHUNK_TOKENS"
    --nerre-model "$NERRE_MODEL"
    --coref-model "$COREF_MODEL"
)

if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip-existing)
fi

python run_pipeline_benchmark.py "${ARGS[@]}"
