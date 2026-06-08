#!/bin/bash

#SBATCH --job-name=uniner-finetune
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --time=36:00:00

echo "Job started on $(date)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo ""

echo "Setting up the environment..."
module purge
module load gnu10
module load python/3.9.9-jh
module load cuda
echo "Loaded modules:"
module list
echo ""

source /scratch/efeldma5/uniner_project/venv/bin/activate
echo "Venv activated: $VIRTUAL_ENV"
echo ""

echo "Starting Python training"
python /scratch/efeldma5/uniner_project/link-kg/scripts/ollama_prep/merge_model.py