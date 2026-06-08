#!/bin/bash
#SBATCH --job-name=coref
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.80gb:1

module load gnu10/10.3.0-ya
module load ollama/0.9.0
module load python/3.9.9-jh

export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
LOG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/runs/ollama_logs"
mkdir -p $OLLAMA_MODELS
mkdir -p "$LOG_DIR"

OLLAMA_LOG="$LOG_DIR/ollama_${SLURM_JOB_ID:-manual}.log"
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
until curl -s http://127.0.0.1:11434 > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is up!"

ollama pull llama3.1:70b

cd /Users/eli/research/link-kg/LinkKG-HS/linkkg
source /scratch/efeldma5/uniner_project/venv/bin/activate

BASE_PREPARED_DIR="/Users/eli/research/link-kg/runs/llama_finetune_2026-03-01_17-02-01/prepared_from_predictions_epoch4"
PROMPTS_DIR="/Users/eli/research/link-kg/LinkKG-HS/linkkg/prompts"
MODEL_NAME="llama3.1:70b"

declare -a ENTITY_CONFIGS=(
    "person person person_nopr"
    "location location location_nopr"
    "organization organization org_nopr"
    "routes routes routes_nopr"
    "means_of_transportation means_of_transportation mot_nopr"
    "smuggled_items smuggled_items smuggleditems_nopr"
)

TOTAL=${#ENTITY_CONFIGS[@]}
CURRENT=0

for CONFIG in "${ENTITY_CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    read -r ENTITY_DIR ENTITY_TYPE PROMPT_PREFIX <<< "$CONFIG"
    
    OUTPUT_DIR="$BASE_PREPARED_DIR/$ENTITY_DIR"
    COREF_PROMPT="$PROMPTS_DIR/${PROMPT_PREFIX}_coref_prompt.txt"
    RESOLVE_PROMPT="$PROMPTS_DIR/${PROMPT_PREFIX}_resolve_prompt.txt"
    
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Processing: $ENTITY_TYPE"
    echo "=========================================="
    echo "Output Directory: $OUTPUT_DIR"
    echo "Coref Prompt: $COREF_PROMPT"
    echo "Resolve Prompt: $RESOLVE_PROMPT"
    echo ""
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "ERROR: Output directory not found: $OUTPUT_DIR"
        continue
    fi
    
    if [ ! -d "$OUTPUT_DIR/chunk_outputs" ] || [ ! -d "$OUTPUT_DIR/ner_outputs" ]; then
        echo "ERROR: Missing chunk_outputs or ner_outputs in $OUTPUT_DIR"
        continue
    fi
    
    if [ ! -f "$COREF_PROMPT" ] || [ ! -f "$RESOLVE_PROMPT" ]; then
        echo "ERROR: Missing prompt files"
        continue
    fi
    
    echo "Starting coref and resolve stages for $ENTITY_TYPE..."
    python run_pipeline.py \
        --input-file-name "${ENTITY_TYPE}_epoch4" \
        --entity-type "$ENTITY_TYPE" \
        --output-dir "$OUTPUT_DIR" \
        --coref-prompt-file "$COREF_PROMPT" \
        --coref-model-name "$MODEL_NAME" \
        --resolve-prompt-file "$RESOLVE_PROMPT" \
        --resolve-model-name "$MODEL_NAME" \
        --run-stages coref resolve
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed $ENTITY_TYPE"
    else
        echo "✗ FAILED for $ENTITY_TYPE"
    fi
    
    echo ""
done