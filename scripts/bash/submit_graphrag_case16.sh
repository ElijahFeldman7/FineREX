#!/bin/bash
#SBATCH --job-name=linkkg-graphrag-16
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
OLLAMA_SKIP_PULL="${OLLAMA_SKIP_PULL:-0}"
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
        --exclude 'runs/' \
        --exclude 'slurm_logs/' \
        --exclude '.git/' \
        "$SOURCE_ROOT"/ "$ROOT"/
fi

PROJECT_ROOT="$ROOT/runs/pipeline_benchmark/$CASE_NAME/graphrag/full_finetuned_all_entities/$CASE_NAME"
INPUT_DIR="$PROJECT_ROOT/input"
mkdir -p "$INPUT_DIR"

SETTINGS_SRC="$ROOT/LinkKG-HS/linkkg/kgconstruction/ragtest/settings.yaml"
PROMPTS_SRC="$ROOT/LinkKG-HS/linkkg/kgconstruction/ragtest/prompts"
if [[ ! -f "$SETTINGS_SRC" ]]; then
    echo "ERROR: Missing settings file at $SETTINGS_SRC"
    exit 1
fi
if [[ ! -d "$PROMPTS_SRC" ]]; then
    echo "ERROR: Missing prompts directory at $PROMPTS_SRC"
    exit 1
fi
cp "$SETTINGS_SRC" "$PROJECT_ROOT/settings.yaml"
mkdir -p "$PROJECT_ROOT/prompts"
rsync -a "$PROMPTS_SRC"/ "$PROJECT_ROOT/prompts"/

RESOLVED_FILE="all_entities_resolved_${CASE_NAME}.txt"
RESOLVED_SRC_SCRATCH="$ROOT/runs/pipeline_benchmark/$CASE_NAME/full_finetuned/$CASE_NAME/all_entities/$RESOLVED_FILE"
RESOLVED_SRC_LOCAL="$SOURCE_ROOT/runs/pipeline_benchmark/$CASE_NAME/full_finetuned/$CASE_NAME/all_entities/$RESOLVED_FILE"

if [[ -f "$RESOLVED_SRC_SCRATCH" ]]; then
    cp "$RESOLVED_SRC_SCRATCH" "$INPUT_DIR/${CASE_NAME}_all_entities_resolved.txt"
elif [[ -f "$RESOLVED_SRC_LOCAL" ]]; then
    cp "$RESOLVED_SRC_LOCAL" "$INPUT_DIR/${CASE_NAME}_all_entities_resolved.txt"
else
    echo "ERROR: Missing resolved text file."
    echo "Checked: $RESOLVED_SRC_SCRATCH"
    echo "Checked: $RESOLVED_SRC_LOCAL"
    exit 1
fi

module load hosts/hopper gnu/12.3.0
module load cuda/12.6.3
module load ollama/0.20.3
module load python/3.12.1-33

export GRAPHRAG_API_KEY="NONE"
export OLLAMA_CONTEXT_LENGTH=16384
export TIKTOKEN_CACHE_DIR=/scratch/efeldma5/uniner_project/tiktoken_cache
export OLLAMA_MODELS=/scratch/efeldma5/uniner_project/ollama_models
export OLLAMA_HOST=127.0.0.1:11434

LOG_DIR="$ROOT/runs/ollama_logs"
OLLAMA_LOG="$LOG_DIR/ollama_graphrag_${SLURM_JOB_ID}.log"
mkdir -p "$ROOT/slurm_logs"
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
    ollama pull llama3.1:70b
    ollama pull nomic-embed-text
fi

source /scratch/efeldma5/uniner_project/venv/bin/activate

pip install -U "graphrag[local]" || pip install -U "graphrag[all]"

python - <<PY
import importlib
import pkgutil
from pathlib import Path

from graphrag.config.enums import IndexingMethod

try:
    from graphrag.index.cli import index_cli
except ModuleNotFoundError:
    from graphrag.cli.index import index_cli

import graphrag_input
import graphrag_cache
import graphrag_storage
import graphrag_vectors

for pkg in (graphrag_input, graphrag_cache, graphrag_storage, graphrag_vectors):
    for module_info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        try:
            importlib.import_module(module_info.name)
        except Exception:
            continue

index_cli(
    root_dir=Path("$PROJECT_ROOT"),
    method=IndexingMethod.Standard,
    verbose=False,
    cache=False,
    dry_run=False,
    skip_validation=True,
)
PY
