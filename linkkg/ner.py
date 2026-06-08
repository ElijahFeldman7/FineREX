import os
import argparse
import json
import requests
from datetime import datetime
from tqdm import tqdm

OLLAMA_URL_TEMPLATE = "http://{host}/api/generate"


def log(msg, log_file=None):
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{time_stamp}] {msg}"
    print(formatted)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted + '\n')


def extract_json_from_ollama(raw_response):
    if not raw_response.strip().startswith("{"):
        raise ValueError("Output does not start with '{'. Unexpected format:\n" + raw_response[:300])

    try:
        result = json.loads(raw_response)

        if not result:
            raise ValueError("Parsed JSON is empty.")

        if "ENTITIES" in result:
            for key in ["PROPER_NOUN", "NOUN_PHRASE"]:
                if key in result["ENTITIES"] and result["ENTITIES"][key] is None:
                    result["ENTITIES"][key] = []

        if "PROPER_NOUN_DESCRIPTION" in result and result["PROPER_NOUN_DESCRIPTION"] is None:
            result["PROPER_NOUN_DESCRIPTION"] = {}

        return result

    except json.JSONDecodeError as e:
        print("\n===== RAW LLM OUTPUT (for debug) =====\n")
        print(raw_response[:1000])
        print("\n======================================\n")
        raise ValueError(f"Invalid JSON format. {e}")


def load_prompt_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def run_ollama_inference(prompt, model_name, ollama_url):
    response = requests.post(ollama_url, json={
        "model": model_name,
        "prompt": prompt,
        "stream": False
    })
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}\n{response.text}")
    return response.json().get('response', '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True, help="Directory of chunk text files")
    parser.add_argument("--prompt-file", required=True, help="NER prompt file")
    parser.add_argument("--output-dir", required=True, help="Directory to save NER results")
    parser.add_argument("--log-file", required=True, help="Path to shared log.txt file")
    parser.add_argument("--model-name", required=True, help="LLM model name")
    parser.add_argument("--max-retries", type=int, default=2, help="Number of retries for invalid JSON output")
    args = parser.parse_args()

    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    ollama_url = OLLAMA_URL_TEMPLATE.format(host=host)

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_template = load_prompt_template(args.prompt_file)

    chunk_files = sorted([f for f in os.listdir(args.chunks_dir) if f.endswith(".txt")])
    log(f"Found {len(chunk_files)} chunk files.", log_file=args.log_file)

    for fname in tqdm(chunk_files, desc="Processing chunks"):
        chunk_path = os.path.join(args.chunks_dir, fname)
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunk_text = f.read()

        base_prompt = f"{prompt_template.strip()}\n\n{chunk_text.strip()}"
        
        print("Base Prompt: ", base_prompt)
        retry_count = 0
        success = False

        while retry_count <= args.max_retries:
            full_prompt = base_prompt
            if retry_count > 0: 
                full_prompt = (
    "IMPORTANT: Your previous output was not valid JSON.\n"
    "- You must return ONLY valid JSON with no explanations.\n"
    "- Each list must contain only quoted strings (e.g., [\"X\", \"Y\"]).\n"
    "- Do NOT include any natural language commentary, reasoning, or interpretation — even inside the list.\n"
    "- The output must strictly match the structure of the example below without deviations.\n"
    '{ "ENTITIES": { "PROPER_NOUN": ["Agent X"], "NOUN_PHRASE": ["the man"] }, "PROPER_NOUN_DESCRIPTION": { "Agent X": "..." } }\n\n'
    "---\n\n"
    + base_prompt
) 

            log(f"Running NER on {fname} (attempt {retry_count + 1})...", log_file=args.log_file)
            
            '''
            try:
                result_raw = run_ollama_inference(full_prompt, args.model_name, ollama_url)
                result = extract_json_from_ollama(result_raw)
                success = True
                break
            except Exception as e:
                log(f"Format issue on attempt {retry_count + 1}: {e}", log_file=args.log_file)
                retry_count += 1
            '''
            try:
                result_raw = run_ollama_inference(full_prompt, args.model_name, ollama_url)
                result = extract_json_from_ollama(result_raw)
                success = True
                break
            except Exception as e:
                log(f"Format issue on attempt {retry_count + 1}: {e}", log_file=args.log_file)

                preview = locals().get("result_raw", "")
                if preview:
                    preview_clean = preview[:300].replace("\n", " ").replace("\r", " ")
                    log(f"[Invalid JSON Preview on attempt {retry_count + 1}] {preview_clean}", log_file=args.log_file)
                retry_count += 1
                if retry_count > args.max_retries:
                    raise RuntimeError(f"Failed after {args.max_retries} attempts for chunk {fname}: {e}")

        raw_out_path = os.path.join(args.output_dir, fname.replace(".txt", "_raw.txt"))
        with open(raw_out_path, 'w', encoding='utf-8') as raw_file:
            raw_file.write(result_raw)

        out_path = os.path.join(args.output_dir, fname.replace(".txt", ".json"))
        with open(out_path, 'w', encoding='utf-8') as out_file:
            json.dump(result, out_file, indent=2)

        log(f"Saved: {out_path}", log_file=args.log_file)


if __name__ == "__main__":
    main()

