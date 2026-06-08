import os
import subprocess
import sys

def process_case(case_name, case_num, entity_types):    
    extraction_csv = "EXTRACTION_CSV_PATH"
    output_base = "OUTPUT_PATH"
    
    if not os.path.exists(extraction_csv):
        print(f"⊘ {case_name}: extraction.csv not found")
        return
    
    prompt_prefixes = {
        "person": "person_nopr",
        "location": "location_nopr",
        "organization": "org_nopr",
        "routes": "routes_nopr",
        "means_of_transportation": "mot_nopr",
        "smuggled_items": "smuggleditems_nopr"
    }
    
    prompts_dir = "prompts"
    
    print(f"\n{'='*60}")
    print(f"Processing {case_name}")
    print(f"{'='*60}")
    
    for entity_type in entity_types:
        print(f"\n  → {entity_type}...", end=" ", flush=True)
        
        prefix = prompt_prefixes.get(entity_type)
        if not prefix:
            print(f"✗ unknown entity type")
            continue
        
        output_dir = os.path.join(output_base, entity_type)
        os.makedirs(output_dir, exist_ok=True)
        
        coref_prompt = os.path.join(prompts_dir, f"{prefix}_coref_prompt.txt")
        resolve_prompt = os.path.join(prompts_dir, f"{prefix}_resolve_prompt.txt")
        
        cmd = [
            "python3", "run_pipeline.py",
            "--input-file-name", case_name,
            "--entity-type", entity_type,
            "--output-dir", output_dir,
            "--input-csv", extraction_csv,
            "--coref-prompt-file", coref_prompt,
            "--coref-model-name", "llama3.1:70b",
            "--resolve-prompt-file", resolve_prompt,
            "--resolve-model-name", "llama3.1:70b",
            "--final-memory-dir", output_base,
            "--run-stages", "prep", "coref", "resolve"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓")
        else:
            print(f"✗")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")

def main():
    cases = [
        ("02USVsYusuf", 2),
        ("13USVsMejOrosco", 13),
        ("15USVsjacquinot", 15)
    ]
    
    entity_types = [
        "person",
        "location",
        "organization",
        "routes",
        "means_of_transportation",
        "smuggled_items"
    ]
    
    for case_name, case_num in cases:
        process_case(case_name, case_num, entity_types)
    
    print(f"\n{'='*60}")
    print("All cases processed. Generating KGs...")
    print(f"{'='*60}\n")
    
    generate_cmd = ["python3", "generate_kgs.py"]
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

if __name__ == "__main__":
    main()
