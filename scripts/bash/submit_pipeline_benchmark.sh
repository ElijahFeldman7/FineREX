#!/bin/bash
#SBATCH --job-name=linkkg-benchmark
#SBATCH --output=/scratch/efeldma5/uniner_project/link-kg/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/efeldma5/uniner_project/link-kg/slurm_logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=72:00:00
#SBATCH --array=0-21

module load hosts/hopper gnu/12.3.0
module load ollama/0.20.3
module load python/3.12.1-33

SOURCE_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "$(basename "$SOURCE_ROOT")" == "setup" ]]; then
    SOURCE_ROOT="$(dirname "$SOURCE_ROOT")"
fi

ROOT="/scratch/efeldma5/uniner_project/link-kg"

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

export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
OLLAMA_PORT=$((11434 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
OLLAMA_URL="http://127.0.0.1:$OLLAMA_PORT/api/generate"
LOG_DIR="$ROOT/runs/ollama_logs"
OLLAMA_LOG="$LOG_DIR/ollama_benchmark_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
mkdir -p $OLLAMA_MODELS
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
until curl -s --fail "http://127.0.0.1:$OLLAMA_PORT/api/tags" > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is up!"

ollama pull llama3.1:70b
ollama pull 2028efeldman/llama-finetuned:latest

source /scratch/efeldma5/uniner_project/venv/bin/activate

cd "$ROOT/LinkKG-HS/linkkg"

mkdir -p "$ROOT/slurm_logs"

OUTPUT_ROOT="$ROOT/runs/pipeline_benchmark/${SLURM_JOB_ID}"

python run_pipeline_benchmark.py \
    --case-index "$SLURM_ARRAY_TASK_ID" \
    --output-root "$OUTPUT_ROOT" \
    --approaches full_70b full_finetuned shortcut \
    --chunk-tokens 300 \
    --ollama-url "$OLLAMA_URL" \
    --skip-existing
