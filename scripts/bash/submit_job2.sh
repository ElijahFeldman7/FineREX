#!/bin/bash
#SBATCH --job-name=linkkg-pipeline
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=12:00:00

module load gnu10/10.3.0-ya
module load ollama/0.9.0
module load python/3.9.9-jh

export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p $OLLAMA_MODELS

ollama serve > /scratch/efeldma5/uniner_project/ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
until curl -s http://127.0.0.1:11434 > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is up!"

ollama pull llama3.1:70b

source /scratch/efeldma5/uniner_project/venv/bin/activate

cd /scratch/efeldma5/uniner_project/link-kg/LinkKG-HS/linkkg

BASE=/scratch/efeldma5/uniner_project/link-kg/runs/llama_finetune_2026-03-01_17-02-01/prepared_from_predictions_epoch4
RUN_ID=pred_epoch4
MODEL='llama3.1:70b'

for t in person location routes means_of_transportation organization smuggled_items; do
    case "$t" in
        means_of_transportation) pfx=mot ;;
        organization)            pfx=org ;;
        smuggled_items)          pfx=smuggleditems ;;
        *)                       pfx="$t" ;;
    esac

    echo "=== Running entity type: $t ==="
    python run_pipeline.py \
        --input-file-name "$RUN_ID" \
        --entity-type "$t" \
        --output-dir "$BASE/$t" \
        --coref-prompt-file "prompts/${pfx}_nopr_coref_prompt.txt" \
        --coref-model-name "$MODEL" \
        --resolve-prompt-file "prompts/${pfx}_nopr_resolve_prompt.txt" \
        --resolve-model-name "$MODEL" \
        --run-stages coref resolve
done

# Cleanup
kill $OLLAMA_PID