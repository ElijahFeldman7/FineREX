#!/bin/bash
#SBATCH --job-name=linkkg-llama70b
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=72:00:00
#SBATCH --array=0-29

module load hosts/hopper gnu/12.3.0
module load ollama/0.20.3
module load python/3.12.1-33

export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
OLLAMA_PORT=$((11434 + SLURM_ARRAY_TASK_ID))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
OLLAMA_URL="http://127.0.0.1:$OLLAMA_PORT/api/generate"
LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/runs/ollama_logs"
OLLAMA_LOG="$LOG_DIR/ollama_${SLURM_JOB_ID}.log"
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

source /scratch/efeldma5/uniner_project/venv/bin/activate

cd /scratch/efeldma5/uniner_project/link-kg/

mkdir -p slurm_logs

python -m scripts.llama_finetune.run_llama70b_splits \
    --split-id "$SLURM_ARRAY_TASK_ID" \
    --ollama-url "$OLLAMA_URL" \
    --skip-existing