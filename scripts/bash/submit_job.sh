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

# Load modules
module load gnu10/10.3.0-ya
module load ollama/0.9.0
module load python/3.9.9-jh

# Setup Ollama
export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p $OLLAMA_MODELS

# Start Ollama server and wait for it to be ready
ollama serve > /scratch/efeldma5/uniner_project/ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
until curl -s http://127.0.0.1:11434 > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is up!"

# Pull model if not already cached
ollama pull llama3.1:70b

# Activate your python env
source /scratch/efeldma5/uniner_project/venv/bin/activate

# Run pipeline
cd /scratch/efeldma5/uniner_project/link-kg/

mkdir -p slurm_logs

python -m scripts.llama_finetune.run_llama70b_splits --split-id "$SLURM_ARRAY_TASK_ID"

kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null || true