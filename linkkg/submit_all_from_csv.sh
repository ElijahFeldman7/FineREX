#!/bin/bash
#SBATCH --job-name=linkkg-csv
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --gres=gpu:A100.80gb:1

mkdir -p slurm_logs

echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"

module purge
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda
module load ollama/0.20.3

OLLAMA=/opt/sw/other/apps/ollama/0.20.3/bin/ollama

export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/runs/ollama_logs"
mkdir -p $OLLAMA_MODELS
mkdir -p "$LOG_DIR"

pkill -u $USER ollama 2>/dev/null || true
sleep 2

$OLLAMA serve > "$LOG_DIR/ollama_${SLURM_JOB_ID:-manual}.log" 2>&1 &
OLLAMA_PID=$!
echo "Ollama started with PID: $OLLAMA_PID"
sleep 5

echo "Waiting for Ollama to start..."
MAX_WAIT=120
ELAPSED=0
until $OLLAMA list > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Ollama did not start after ${MAX_WAIT}s. Last log output:"
        tail -20 "$LOG_DIR/ollama_${SLURM_JOB_ID:-manual}.log"
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo "Ollama is up!"

$OLLAMA pull llama3.1:70b

source /scratch/efeldma5/uniner_project/venv/bin/activate

BASE_DIR="/scratch/efeldma5/uniner_project/link-kg/runs/eval_llama_finetuned/2026-04-25_19-50-47"
OUTPUT_BASE_DIR="/scratch/efeldma5/uniner_project/link-kg/runs/eval_llama_finetuned/2026-04-25_19-50-47/processed_kg"
MODEL_NAME="llama3.1:70b"

cd /scratch/efeldma5/uniner_project/link-kg/LinkKG-HS/linkkg

ENTITY_TYPES=("person" "location" "organization" "routes" "means_of_transportation" "smuggled_items")

for ENTITY_TYPE in "${ENTITY_TYPES[@]}"; do    
    python run_all_cases_from_csv.py \
        --base-dir "$BASE_DIR" \
        --entity-type "$ENTITY_TYPE" \
        --model-name "$MODEL_NAME" \
        --output-base-dir "$OUTPUT_BASE_DIR"
        
    echo "Finished processing $ENTITY_TYPE"
    echo ""
done

echo "All tasks completed on $(date)"

kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null || true
echo "Ollama stopped."
