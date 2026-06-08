import os
import re
import csv
import json

INPUT_ROOT = "/Users/eli/research/link-kg/runs/eval_llama_finetuned"
OUTPUT_ROOT = "/Users/eli/research/link-kg/runs/eval_llama_finetuned/formatted"


def parse_entity_tuples(output_text: str) -> dict:
    entities = {}
    proper_noun_descriptions = {}

    entity_pattern = re.compile(
        r'\("?entity"?\|([^|]+)\|([^|]+)\|([^)]+)\)',
        re.IGNORECASE
    )

    for match in entity_pattern.finditer(output_text):
        name = match.group(1).strip()
        etype = match.group(2).strip()
        description = match.group(3).strip()

        entities[name] = {
            "type": etype,
            "description": description
        }
        proper_noun_descriptions[name] = description

    return {
        "ENTITIES": entities,
        "PROPER_NOUN_DESCRIPTION": proper_noun_descriptions
    }


def convert_case(csv_path: str, output_dir: str, case_name: str) -> None:
    chunks_dir = os.path.join(output_dir, "chunk_outputs")
    ner_dir = os.path.join(output_dir, "ner_outputs")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(ner_dir, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  [{case_name}] {len(rows)} chunks found")

    for row in rows:
        chunk_num = row["chunk_number"].strip().zfill(4)
        input_text = row["input_text"].strip()
        output_text = row["output_text"].strip()

        chunk_name = f"chunk_{chunk_num}"

        chunk_path = os.path.join(chunks_dir, chunk_name + ".txt")
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(input_text)

        ner_data = parse_entity_tuples(output_text)
        ner_path = os.path.join(ner_dir, chunk_name + ".json")
        with open(ner_path, "w", encoding="utf-8") as f:
            json.dump(ner_data, f, indent=2)

    print(f"  [{case_name}] Done -> {output_dir}")


def main():
    # Find all timestamp folders, then case subfolders inside each
    all_cases = []  # list of (csv_path, case_name)

    timestamp_dirs = sorted([
        d for d in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
        and d != "formatted"
    ])

    for ts_dir in timestamp_dirs:
        ts_path = os.path.join(INPUT_ROOT, ts_dir)
        case_subdirs = sorted([
            d for d in os.listdir(ts_path)
            if os.path.isdir(os.path.join(ts_path, d))
        ])
        for case_name in case_subdirs:
            csv_path = os.path.join(ts_path, case_name, "extraction.csv")
            if os.path.exists(csv_path):
                all_cases.append((csv_path, case_name))

    if not all_cases:
        raise RuntimeError(f"No extraction.csv files found under {INPUT_ROOT}")

    print(f"Found {len(all_cases)} cases to process\n")

    for i, (csv_path, case_name) in enumerate(all_cases, start=1):
        output_dir = os.path.join(OUTPUT_ROOT, case_name)
        print(f"[{i}/{len(all_cases)}] {case_name}")
        convert_case(csv_path, output_dir, case_name)

    print(f"\nAll cases written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()