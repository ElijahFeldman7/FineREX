import argparse
import csv
import json
import os
import re
from typing import Dict, List, Tuple


ENTITY_TUPLE_PATTERN = re.compile(
    r'\(\s*"?entity"?\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\)',
    re.IGNORECASE | re.DOTALL,
)


def normalize_entity_type(value: str) -> str:
    return value.strip().lower().replace("_", " ").replace("-", " ").strip()


def clean_field(value: str) -> str:
    return value.strip().strip('"').strip()


def parse_entities(predicted_text: str, requested_entity_type: str) -> List[Tuple[str, str]]:
    matches = ENTITY_TUPLE_PATTERN.findall(predicted_text or "")
    wanted = normalize_entity_type(requested_entity_type)
    entities: List[Tuple[str, str]] = []

    for name, etype, description in matches:
        if normalize_entity_type(etype) != wanted:
            continue
        entity_name = clean_field(name)
        entity_desc = clean_field(description)
        if entity_name:
            entities.append((entity_name, entity_desc))

    return entities


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build LinkKG chunk_outputs and ner_outputs from predictions CSV so coref/resolve "
            "can run without re-running NER."
        )
    )
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--entity-type", required=True)
    parser.add_argument("--text-column", default="Input_Text")
    parser.add_argument("--prediction-column", default="Predicted_Text")
    args = parser.parse_args()

    chunk_dir = os.path.join(args.output_dir, "chunk_outputs")
    ner_dir = os.path.join(args.output_dir, "ner_outputs")
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(ner_dir, exist_ok=True)

    rows_written = 0
    rows_with_entities = 0
    summary: Dict[str, int] = {
        "rows_total": 0,
        "rows_written": 0,
        "rows_with_entities": 0,
        "entities_written": 0,
    }

    with open(args.predictions_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            summary["rows_total"] += 1
            text = (row.get(args.text_column) or "").strip()
            predicted = row.get(args.prediction_column) or ""

            if not text:
                continue

            chunk_name = f"chunk_{idx:02d}"
            chunk_path = os.path.join(chunk_dir, f"{chunk_name}.txt")
            ner_path = os.path.join(ner_dir, f"{chunk_name}.json")

            with open(chunk_path, "w", encoding="utf-8") as chunk_file:
                chunk_file.write(text)

            entities = parse_entities(predicted, args.entity_type)
            names: List[str] = []
            desc_map: Dict[str, str] = {}

            for name, desc in entities:
                if name not in desc_map:
                    names.append(name)
                    desc_map[name] = desc

            payload = {
                "ENTITIES": {
                    "PROPER_NOUN": names,
                    "NOUN_PHRASE": [],
                },
                "PROPER_NOUN_DESCRIPTION": desc_map,
            }

            with open(ner_path, "w", encoding="utf-8") as ner_file:
                json.dump(payload, ner_file, ensure_ascii=True, indent=2)

            rows_written += 1
            summary["entities_written"] += len(names)
            if names:
                rows_with_entities += 1

    summary["rows_written"] = rows_written
    summary["rows_with_entities"] = rows_with_entities

    summary_path = os.path.join(args.output_dir, "prepared_from_predictions_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=True, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Prepared inputs in: {args.output_dir}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()