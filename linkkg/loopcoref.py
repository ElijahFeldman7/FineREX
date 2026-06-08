import os
import argparse
import json
import requests
from datetime import datetime

def log(msg, log_file_path=None):
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{time_stamp}] {msg}"
    print(full_msg)
    if log_file_path:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(full_msg + "\n")

def extract_json_from_ollama(raw_response):
    cleaned = raw_response.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Output does not contain JSON object. Unexpected format:\n{cleaned[:200]}")
    if start != 0 or end != len(cleaned) - 1:
        cleaned = cleaned[start:end + 1]

    try:
        result = json.loads(cleaned)
        if "RESOLVED_ENTITIES" not in result or "AUXILIARY_DESCRIPTIONS" not in result:
            raise ValueError("Valid JSON but missing required keys.")
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format. {e}")

def load_prompt_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def run_ollama_inference(prompt, model, url):
    response = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "num_predict": 1000
        }
    }, timeout=600)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}\n{response.text}")
    return response.json()['response']

def inject_prompt(prompt_template, resolved_entities, aux_descriptions, ner_entities, aux_current, chunk_text, verification=False):
    payload = {
        "RESOLVED_ENTITIES": resolved_entities,
        "AUXILIARY_DESCRIPTIONS": aux_descriptions,
        "ENTITIES": ner_entities,
        "AUX_DESC": aux_current,
        "CHUNK_TEXT": chunk_text
    }
    return prompt_template.strip() + "\n\n" + json.dumps(payload, indent=2)

def process_chunks(chunk_files, args, prompt_template, resolved_entities, aux_descriptions, verification=False):
    for fname in chunk_files:
        if not fname.endswith(".txt"):
            continue

        chunk_name = fname.replace(".txt", "")
        chunk_path = os.path.join(args.chunks_dir, fname)
        ner_path = os.path.join(args.ner_dir, chunk_name + ".json")

        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_text = f.read()
        with open(ner_path, 'r', encoding='utf-8') as f:
            ner_data = json.load(f)

        ner_entities = ner_data.get("ENTITIES", {})
        aux_current = ner_data.get("PROPER_NOUN_DESCRIPTION", {})

        full_prompt = inject_prompt(
            prompt_template,
            resolved_entities,
            aux_descriptions,
            ner_entities,
            aux_current,
            chunk_text,
            verification=verification
        )
        log(f"Processing {chunk_name} (verification={verification})...", args.log_file)
        
        '''
        try:
            result_raw = run_ollama_inference(full_prompt, args.model, args.ollama_url)
            result = extract_json_from_ollama(result_raw)
        except Exception as e:
            log(f"Failed to process {chunk_name}: {e}", args.log_file)
            raise
        '''
        
        max_retries = max(1, args.max_retries if hasattr(args, "max_retries") else 3)
        attempt = 0
        success = False
        last_error = None
        while attempt < max_retries and not success:
            try:
                result_raw = run_ollama_inference(full_prompt, args.model, args.ollama_url)
                try:
                    result = extract_json_from_ollama(result_raw)
                    success = True
                except Exception as json_error:
                    log(f"[Attempt {attempt+1}] JSON parsing failed for {chunk_name}. Raw model output:\n{result_raw}", args.log_file)
                    raise json_error  # Let it go to outer except to handle retries
            except Exception as e:
                attempt += 1
                last_error = e
                log(f"[Attempt {attempt}] Failed to process {chunk_name}: {e}", args.log_file)
                if attempt == max_retries:
                    log(f"Maximum retries reached for {chunk_name}. Skipping.", args.log_file)

        if not success:
            error_text = str(last_error) if last_error else "Unknown error"
            result = {
                "RESOLVED_ENTITIES": {},
                "AUXILIARY_DESCRIPTIONS": {},
                "ERROR": error_text
            }
        else:
            resolved_entities.update(result.get("RESOLVED_ENTITIES", {}))
            aux_descriptions.update(result.get("AUXILIARY_DESCRIPTIONS", {}))

        out_path = os.path.join(args.output_dir, chunk_name + ("_verify" if verification else "") + ".json")
        with open(out_path, 'w', encoding='utf-8') as out_file:
            json.dump(result, out_file, indent=2)

        log(f"Saved coref output: {out_path}", args.log_file)

    final_memory_path = os.path.join(args.base_output_folder, "final_memory.json")
    with open(final_memory_path, 'w', encoding='utf-8') as f:
        json.dump({
            "RESOLVED_ENTITIES": resolved_entities,
            "AUXILIARY_DESCRIPTIONS": aux_descriptions
        }, f, indent=2)
    log(f"Final memory saved to: {final_memory_path}", args.log_file)

def run_loopcoref(
    chunks_dir,
    ner_dir,
    prompt_file,
    base_output_folder,
    input_file_name,
    model="llama3370gb32k",
    verify_passes=0,
    log_file=None,
    verify_prompt_file=None,
    max_retries=3
):
    # Set Ollama URL here
    ollama_url = f"http://{os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')}/api/generate"

    output_path = os.path.join(base_output_folder, "coref_outputs")
    os.makedirs(output_path, exist_ok=True)

    discovery_prompt = load_prompt_template(prompt_file)
    verify_prompt = load_prompt_template(verify_prompt_file) if verify_prompt_file else discovery_prompt
    chunk_files = sorted(os.listdir(chunks_dir))

    resolved_entities = {}
    aux_descriptions = {}

    # Create a dummy args-like object for process_chunks compatibility or refactor process_chunks
    class Args:
        pass
    args = Args()
    args.chunks_dir = chunks_dir
    args.ner_dir = ner_dir
    args.log_file = log_file
    args.model = model
    args.ollama_url = ollama_url
    args.output_dir = output_path
    args.base_output_folder = base_output_folder
    args.max_retries = max_retries

    process_chunks(chunk_files, args, discovery_prompt, resolved_entities, aux_descriptions, verification=False)

    for i in range(verify_passes):
        log(f"Starting verification pass {i+1}...", log_file)
        process_chunks(chunk_files, args, verify_prompt, resolved_entities, aux_descriptions, verification=True)

    return resolved_entities, aux_descriptions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--ner-dir", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--verify-prompt-file")
    parser.add_argument("--base-output-folder", required=True)
    parser.add_argument("--input-file-name", required=True)
    parser.add_argument("--model", default="llama3370gb32k")
    parser.add_argument("--verify-passes", type=int, default=0)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for failed coref requests")
    args = parser.parse_args()

    run_loopcoref(
        chunks_dir=args.chunks_dir,
        ner_dir=args.ner_dir,
        prompt_file=args.prompt_file,
        base_output_folder=args.base_output_folder,
        input_file_name=args.input_file_name,
        model=args.model,
        verify_passes=args.verify_passes,
        log_file=args.log_file,
        verify_prompt_file=args.verify_prompt_file,
        max_retries=args.max_retries
    )
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()

