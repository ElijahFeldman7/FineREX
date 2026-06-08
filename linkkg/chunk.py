import argparse
import re
import os
import json
from datetime import datetime
import time
log_file = None  # Will be set dynamically

def log(msg):
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{time_stamp}] {msg}"
    print(full_msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(full_msg + '\n')

def chunk_legal_text(text, max_tokens=2000, tokenizer=None, min_last_chunk_words=20):
    paragraphs = re.split(r'\n{2,}', text.strip())
    chunks = []
    current_chunk = []
    current_length = 0
    #log(f"Number of paragraphs: {len(paragraphs)}")
    #log(f"First paragraph (truncated): {paragraphs[0][:100] if paragraphs else 'None'}")
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        token_count = len(tokenizer.encode(para)) if tokenizer else len(para.split())

        if current_length + token_count > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = token_count
        else:
            current_chunk.append(para)
            current_length += token_count
    print(chunks)
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    if len(chunks) >= 2:
        last_chunk_words = len(chunks[-1].split())
        if last_chunk_words < min_last_chunk_words:
            log(f"Last chunk too small ({last_chunk_words} words). Merging with previous.")
            chunks[-2] += "\n\n" + chunks[-1]
            chunks.pop()

    return chunks

def main():
    global log_file
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Chunk legal case text by tokens.")
    parser.add_argument("--input-file", required=True, help="Path to the input .txt file")
    parser.add_argument("--output-dir", required=True, help="Directory to save chunked files")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens per chunk")
    parser.add_argument("--min-last-chunk-words", type=int, default=20, help="Minimum words for last chunk before merging")
    parser.add_argument("--use-tokenizer", action="store_true", help="Use GPT-2 tokenizer for token count")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(os.path.dirname(args.output_dir), "log.txt")
    log(f"Starting chunking for {args.input_file}...")

    tokenizer = None
    if args.use_tokenizer:
        try:
            from transformers import AutoTokenizer
            log("Loading GPT-2 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except ImportError:
            log("Transformers library is not installed. Run: pip install transformers")
            return

    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = chunk_legal_text(
        text,
        max_tokens=args.max_tokens,
        tokenizer=tokenizer,
        min_last_chunk_words=args.min_last_chunk_words
    )

    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(args.output_dir, f"chunk_{i+1:02d}.txt")  # zero-padded to 2 digits
        with open(chunk_path, 'w', encoding='utf-8') as out_file:
            out_file.write(chunk)
        log(f"Saved chunk {i+1:02d} ({len(chunk.split())} words)")

    end_time = time.time()
    log(f"Successfully created {len(chunks)} chunks in '{args.output_dir}'.")
    log(f"Total time taken: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
