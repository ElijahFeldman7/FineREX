<div align="center"><h1>FineREX: Fine-Tuned NER-RE for Human Smuggling Knowledge Graphs</h1></div>

## Abstract
Court proceedings contain valuable evidence about human smuggling networks, but this information is often buried within unstructured, jargon-heavy legal documents. While large language models (LLMs) can support knowledge graph construction through automated information extraction, existing approaches rely on general-purpose models that are not tailored to the entity and relationship definitions required in this domain. We introduce FineREX, a streamlined knowledge graph construction pipeline built around a fine-tuned LLM for named entity recognition and relationship extraction (NER-RE). Using a manually annotated dataset of   text chunks, FineREX achieves absolute improvements of 15.50% and 31.46% in entity and relationship F1-score, respectively, compared to a larger general-purpose baseline. These gains translate into higher-quality knowledge graphs, reducing legal noise by nearly half and lowering node duplication on long documents from 17.78% to 11.17%. By eliminating document rewriting and redundant extraction stages, FineREX also reduces end-to-end processing time by 50.0%. Our results demonstrate that domain-specific fine-tuning can substantially outperform larger general-purpose models while improving both the quality and efficiency of knowledge graph construction for illicit network analysis. 

FineREX is a framework for domain-specific knowledge graph construction, specializing in human smuggling. It introduces a fine-tuned approach for Named Entity Recognition and Relationship Extraction (NER-RE) using Llama 3.1 models, outperforming generic baseline pipelines.

Our fine-tuned model may be accessed at [Model](https://ollama.com/2028efeldman/llama-finetuned), and run with ollama run 2028efeldman/llama-finetuned.

## Project Structure

### FineREX (Core Implementation)
The `scripts/` directory contains the implementation of the FineREX approach:
- `scripts/llama_finetune/`: Core training and inference logic.
  - `train.py`: QLoRA-based fine-tuning for NER-RE.
  - `run_splits.py`: Batch inference using fine-tuned models.
- `scripts/kg/`: Knowledge Graph construction and consolidation.
  - `build_consolidated_kg_networkx.py`: The primary FineREX script for merging model extractions into canonicalized KGs using NetworkX.
  - `build_eval_case_kgs.py`: Generates KGs for specific evaluation cases.
- `scripts/util/`: Utilities for bridging model outputs to graph processing.

### LinkKG (Baseline)
The `linkkg/` directory contains the baseline pipeline used for comparison:
- `linkkg/run_pipeline.py`: A modular pipeline stage manager.
- `linkkg/ner.py`, `linkkg/loopcoref.py`, `linkkg/resolve_coref.py`: Baseline stages using non-specialized or 70B models.
- `linkkg/generate_kgs.py`: Simple KG generation for baseline results.

### Data and Guidelines
- `dataset/guidelines.md`: Comprehensive annotation guidelines and entity definitions (PERSON, LOCATION, ORGANIZATION, etc.).
- `setup/requirements.txt`: Environment dependencies.

## Installation

### Prerequisites
- Python 3.12
- CUDA-compatible GPU

### Setup
We recommend using `uv` for environment management:

```bash
uv venv --python 3.12
source .venv/bin/activate
pip install -r setup/requirements.txt
```
## FineREX Pipeline

### Fine-tuning Process
Configure `scripts/llama_finetune/config.py` with your dataset paths and base model, then run:
```bash
python -m scripts.llama_finetune.train
```

For evaluation and NER-RE extraction of this model, use:
```bash
python -m scripts.llama_finetune.run_llama8b_splits --output-root runs/finetune_results
```

### Coreference Resolution
```bash
bash scripts/util/run_coref_finetune.sh
```

### KG Consolidation
Construct the final canonicalized Knowledge Graph from the model's metrics/predictions:
```bash
python scripts/kg/build_consolidated_kg_networkx.py
```

## Baseline Comparison (LinkKG)
To run the baseline pipeline for comparison:
```bash
python linkkg/run_pipeline.py \
    --input-file-name "case_name" \
    --entity-type "PERSON" \
    --run-stages prep chunk ner coref resolve \
    --ner-model-name "llama3.1:70b" \
    --ner-prompt-file "linkkg/prompts/person_nopr_ner_prompt.txt"
```

## Citation

```bibtex
@article{feldmanfinerex2026,
  title   = {{FineREX}: Fine-Tuned {NER-RE} for Human Smuggling Knowledge Graphs},
  author  = {Feldman, Elijah and Meher, Dipak and Domeniconi, Carlotta},
  journal = {arXiv preprint arXiv:2606.19710},
  year    = {2026},
  url     = {[https://arxiv.org/abs/2606.19710](https://arxiv.org/abs/2606.19710)}
}
