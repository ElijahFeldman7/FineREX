<div align="center"><h1>FineREX: Fine-Tuned NER-RE for Human Smuggling Knowledge Graphs</h1></div>

FineREX is a framework for extracting named entities and relationships from legal case documents related to human smuggling. It uses fine-tuned Large Language Models (LLMs) to construct high-quality Knowledge Graphs (KGs) specialized for the human smuggling domain.

## Overview

The project provides a complete pipeline for domain-specific knowledge extraction:
1.  **Dataset Preparation**: Guidelines for annotating human smuggling entities and relationships in legal texts.
2.  **Model Fine-tuning**: QLoRA-based fine-tuning of Llama 3.1 models for specialized NER-RE.
3.  **KG Construction**: A multi-stage pipeline (LinkKG) including chunking, NER extraction, coreference resolution, and final entity resolution.
4.  **Evaluation**: Tools for comparing constructed KGs and calculating statistical metrics.

## Project Structure

- `dataset/`: Contains `guidelines.md` with detailed annotation rules and entity definitions.
- `linkkg/`: Core pipeline for knowledge graph construction (LinkKG).
  - `run_pipeline.py`: Main entry point for the extraction pipeline.
  - `ner.py`, `loopcoref.py`, `resolve_coref.py`: Individual pipeline stages.
  - `kgconstruction/`: Integration with GraphRAG and other KG tools.
  - `prompts/`: Domain-specific prompts for different entity types and pipeline stages.
- `scripts/`:
  - `llama_finetune/`: Scripts for fine-tuning Llama models using QLoRA.
  - `baseline/`: Baseline extraction scripts using vanilla Llama models.
  - `kg/`: Utilities for building and rendering consolidated KGs.
  - `bash/`: Shell scripts for running batch jobs and benchmarks.
  - `statistics/`: Scripts for statistical analysis of extraction results.
- `setup/`: Environment configuration and requirements.

## Installation

### Prerequisites
- Python 3.12
- CUDA-compatible GPU

### Setup

We recommend using `uv` for environment management, but standard `pip` can also be used.

1.  **Create a virtual environment**:
    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r setup/requirements.txt
    ```

## Usage

### KG Construction Pipeline
The main pipeline is located in `linkkg/run_pipeline.py`. It supports stages: `prep`, `chunk`, `ner`, `coref`, and `resolve`.

```bash
python linkkg/run_pipeline.py \
    --input-file-name "case_name" \
    --entity-type "PERSON" \
    --input-file "path/to/input.txt" \
    --run-stages prep chunk ner coref resolve \
    --ner-model-name "your-model-name" \
    --ner-prompt-file "linkkg/prompts/person_nopr_ner_prompt.txt" \
    --coref-model-name "your-model-name" \
    --coref-prompt-file "linkkg/prompts/person_nopr_coref_prompt.txt" \
    --resolve-model-name "your-model-name" \
    --resolve-prompt-file "linkkg/prompts/person_nopr_resolve_prompt.txt"
```

### Model Fine-tuning
Fine-tuning scripts are located in `scripts/llama_finetune/`. Configure the model and paths in `config.py` before running.

```bash
python -m scripts.llama_finetune.train
```

## Entity Types
FineREX specializes in extracting the following 7 entity types:
1.  **PERSON**: Individuals such as smugglers, migrants, and border agents.
2.  **LOCATION**: Geographical names (cities, countries, states).
3.  **ORGANIZATION**: Smuggling rings, cartels, or companies.
4.  **MEANS_OF_TRANSPORTATION**: Vehicles used for transport (cars, trucks, etc.).
5.  **MEANS_OF_COMMUNICATION**: Tools for coordination (phones, WhatsApp, etc.).
6.  **ROUTES**: Specific roads, highways, or desert paths.
7.  **SMUGGLED_ITEMS**: Contraband, weapons, or other illegal goods.

Detailed definitions and extraction rules are available in `dataset/guidelines.md`.
