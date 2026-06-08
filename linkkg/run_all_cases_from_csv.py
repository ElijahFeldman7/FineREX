import os
import argparse
import subprocess
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, help="Base directory containing case folders (e.g., runs/eval_llama_finetuned/2026-04-25_19-50-47)")
    parser.add_argument("--entity-type", required=True, help="Entity type to process")
    parser.add_argument("--model-name", default="llama3.1:70b", help="Model name for coref and resolve")
    parser.add_argument("--output-base-dir", default="output_from_csv", help="Base directory for outputs")
    args = parser.parse_args()

    cases = sorted([d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))])
    print(f"Found {len(cases)} cases.")

    prompts_dir = "prompts"
    prompt_prefixes = {
        "person": "person_nopr",
        "location": "location_nopr",
        "organization": "org_nopr",
        "routes": "routes_nopr",
        "mot": "mot_nopr",
        "means_of_transportation": "mot_nopr",
        "smuggleditems": "smuggleditems_nopr",
        "smuggled_items": "smuggleditems_nopr",
        "moc": "moc_nopr",
        "means_of_communication": "moc_nopr"
    }

    entity_key = args.entity_type.lower()
    if entity_key not in prompt_prefixes:
        print(f"Error: Unknown entity type {args.entity_type}")
        return

    prefix = prompt_prefixes[entity_key]
    coref_prompt = os.path.join(prompts_dir, f"{prefix}_coref_prompt.txt")
    resolve_prompt = os.path.join(prompts_dir, f"{prefix}_resolve_prompt.txt")

    for case in cases:
        csv_file = os.path.join(args.base_dir, case, "extraction.csv")
        if not os.path.exists(csv_file):
            print(f"Skipping {case}: extraction.csv not found.")
            continue

        output_dir = os.path.join(args.output_base_dir, case, args.entity_type)
        os.makedirs(output_dir, exist_ok=True)

        final_memory_dir = os.path.join("/Users/eli/research/link-kg/datasets/processed_kg", case)

        print(f"\n>>> Processing case: {case} for entity: {args.entity_type}")
        
        cmd = [
            "python", "run_pipeline.py",
            "--input-file-name", case,
            "--entity-type", args.entity_type,
            "--output-dir", output_dir,
            "--input-csv", csv_file,
            "--coref-prompt-file", coref_prompt,
            "--coref-model-name", args.model_name,
            "--resolve-prompt-file", resolve_prompt,
            "--resolve-model-name", args.model_name,
            "--final-memory-dir", final_memory_dir,
            "--run-stages", "prep", "coref", "resolve"
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
