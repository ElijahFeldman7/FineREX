import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate resolved KGs from final_memory.json files"
    )
    parser.add_argument(
        "--datasets-dir",
        default="/Users/eli/research/link-kg/datasets/processed_kg",
        help="Path to datasets/processed_kg directory"
    )
    parser.add_argument(
        "--model-name",
        default="llama3.1:70b",
        help="Model name for resolve stage"
    )
    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    entity_types = [
        "person",
        "location",
        "organization",
        "routes",
        "means_of_transportation",
        "smuggled_items"
    ]

    prompt_prefixes = {
        "person": "person_nopr",
        "location": "location_nopr",
        "organization": "org_nopr",
        "routes": "routes_nopr",
        "means_of_transportation": "mot_nopr",
        "smuggled_items": "smuggleditems_nopr"
    }

    prompts_dir = "prompts"

    # Get all cases
    cases = sorted([
        d for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d))
    ])

    print(f"Found {len(cases)} cases to regenerate")

    total = len(cases) * len(entity_types)
    completed = 0

    for case in cases:
        case_path = os.path.join(datasets_dir, case)

        for entity_type in entity_types:
            entity_path = os.path.join(case_path, entity_type)
            
            if not os.path.exists(entity_path):
                print(f"  ⊘ {case} / {entity_type}: directory not found, skipping")
                completed += 1
                continue

            chunk_outputs = os.path.join(entity_path, "chunk_outputs")
            final_memory = os.path.join(entity_path, "final_memory.json")

            if not os.path.exists(chunk_outputs):
                print(f"  ⊘ {case} / {entity_type}: chunk_outputs not found, skipping")
                completed += 1
                continue

            if not os.path.exists(final_memory):
                print(f"  ⊘ {case} / {entity_type}: final_memory.json not found, skipping")
                completed += 1
                continue

            prefix = prompt_prefixes.get(entity_type)
            if not prefix:
                print(f"  ⊘ {case} / {entity_type}: unknown entity type, skipping")
                completed += 1
                continue

            resolve_prompt = os.path.join(prompts_dir, f"{prefix}_resolve_prompt.txt")
            if not os.path.exists(resolve_prompt):
                print(f"  ⊘ {case} / {entity_type}: resolve prompt not found at {resolve_prompt}, skipping")
                completed += 1
                continue

            print(f"  → [{completed+1}/{total}] Regenerating {case} / {entity_type}...", end=" ", flush=True)

            log_file = os.path.join(entity_path, "log.txt")

            cmd = [
                sys.executable, "resolve_coref.py",
                "--chunks-dir", chunk_outputs,
                "--final-memory", final_memory,
                "--prompt-file", resolve_prompt,
                "--base-output-dir", entity_path,
                "--input-file-name", case,
                "--model-name", args.model_name,
                "--num-retries", "1",
                "--num-ctx", "8192",
                "--request-timeout", "600",
                "--log-file", log_file,
                "--entity-type", entity_type
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✓")
            else:
                print(f"✗\n    Error: {result.stderr}")

            completed += 1

    print(f"\nRegeneration complete: {completed}/{total} processed")

if __name__ == "__main__":
    main()
