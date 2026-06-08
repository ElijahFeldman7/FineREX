#!/bin/bash
#SBATCH --job-name=coref
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

mkdir -p /scratch/efeldma5/uniner_project/logs

echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"

# Load modules
module purge
module load gnu10/10.3.0-ya
module load python/3.9.9-jh
module load cuda

# Hardcoded ollama binary
OLLAMA=/opt/sw/other/apps/ollama/0.20.3/bin/ollama

# Setup Ollama
export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/runs/ollama_logs"
mkdir -p $OLLAMA_MODELS
mkdir -p "$LOG_DIR"

# Kill any leftover Ollama processes
pkill -u $USER ollama 2>/dev/null || true
sleep 2

# Start Ollama
OLLAMA_LOG="$LOG_DIR/ollama_${SLURM_JOB_ID:-manual}.log"
$OLLAMA serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
echo "Ollama started with PID: $OLLAMA_PID (log: $OLLAMA_LOG)"
sleep 5

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
MAX_WAIT=120
ELAPSED=0
until $OLLAMA list > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Ollama did not start after ${MAX_WAIT}s. Last log output:"
        tail -20 "$OLLAMA_LOG"
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo "  ...still waiting (${ELAPSED}s elapsed)"
done
echo "Ollama is up!"

# Pull model
$OLLAMA pull llama3.1:70b

# Activate venv
source /scratch/efeldma5/uniner_project/venv/bin/activate

# Paths
FORMATTED_DIR="/scratch/efeldma5/uniner_project/link-kg/runs/eval_llama_finetuned/formatted"
PROMPTS_DIR="/scratch/efeldma5/uniner_project/link-kg/LinkKG-HS/linkkg/prompts"
MODEL_NAME="llama3.1:70b"

cd /scratch/efeldma5/uniner_project/link-kg/LinkKG-HS/linkkg

declare -a ENTITY_CONFIGS=(
    "person person_nopr"
    "location location_nopr"
    "organization org_nopr"
    "routes routes_nopr"
    "means_of_transportation mot_nopr"
    "smuggled_items smuggleditems_nopr"
)

CASE_DIRS=($(ls -d $FORMATTED_DIR/*/))
TOTAL_CASES=${#CASE_DIRS[@]}
TOTAL_ENTITY_TYPES=${#ENTITY_CONFIGS[@]}

echo "Found $TOTAL_CASES cases"
echo "Processing $TOTAL_ENTITY_TYPES entity types per case"
echo ""

CASE_NUM=0
for CASE_PATH in "${CASE_DIRS[@]}"; do
    CASE_NUM=$((CASE_NUM + 1))
    CASE_NAME=$(basename "$CASE_PATH")

    echo "=========================================="
    echo "[$CASE_NUM/$TOTAL_CASES] Case: $CASE_NAME"
    echo "=========================================="

    if [ ! -d "$CASE_PATH/chunk_outputs" ] || [ ! -d "$CASE_PATH/ner_outputs" ]; then
        echo "ERROR: Missing chunk_outputs or ner_outputs in $CASE_PATH — skipping"
        continue
    fi

    ENTITY_NUM=0
    for CONFIG in "${ENTITY_CONFIGS[@]}"; do
        ENTITY_NUM=$((ENTITY_NUM + 1))
        read -r ENTITY_TYPE PROMPT_PREFIX <<< "$CONFIG"

        COREF_PROMPT="$PROMPTS_DIR/${PROMPT_PREFIX}_coref_prompt.txt"
        RESOLVE_PROMPT="$PROMPTS_DIR/${PROMPT_PREFIX}_resolve_prompt.txt"
        echo "  [$ENTITY_NUM/$TOTAL_ENTITY_TYPES] $ENTITY_TYPE"

            if [ ! -f "$COREF_PROMPT" ] || [ ! -f "$RESOLVE_PROMPT" ]; then
                echo "  ERROR: Missing prompt files for $ENTITY_TYPE — skipping"
                continue
            fi

            # We pass $CASE_PATH as the output-dir because run_pipeline.py 
            # automatically looks for chunk_outputs and ner_outputs inside it.
            python run_pipeline.py \
                --input-file-name "${CASE_NAME}_${ENTITY_TYPE}" \
                --entity-type "$ENTITY_TYPE" \
                --output-dir "$CASE_PATH" \
                --coref-prompt-file "$COREF_PROMPT" \
                --coref-model-name "$MODEL_NAME" \
                --resolve-prompt-file "$RESOLVE_PROMPT" \
                --resolve-model-name "$MODEL_NAME" \
                --run-stages coref resolve
        if [ $? -eq 0 ]; then
            echo "  ✓ $ENTITY_TYPE done"
        else
            echo "  ✗ $ENTITY_TYPE FAILED"
        fi
    done

    echo ""
done

echo "All cases completed on $(date)"

# Cleanup
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null || true
echo "Ollama stopped."