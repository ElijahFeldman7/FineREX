import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import getpass
import socket

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, set_seed

from .config import (
    BASE_MODEL_NAME,
    DATASET_PATH,
    MAX_LENGTH,
    RUN_NAME,
    format_system_prompt,
)
from .data_processing import create_preprocess_function
from .model_setup import setup_model_and_tokenizer, setup_peft_model
from .metrics import compute_metrics_wrapper, preprocess_logits_for_metrics
from .trainer import CustomTrainer

INSTRUCTION_TEMPLATE = """Input_text: \n{input_text}\nOutput:\n"""


@dataclass
class SplitManifest:
    split_id: int
    seed: int
    dataset_path: str
    dataset_sha256: str
    total_rows: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    train_row_ids: list
    val_row_ids: list
    test_row_ids: list


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_clean_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).fillna("")
    df["row_id"] = df.index
    df = df.dropna(subset=["Input_Text", "Output"])
    df = df[df["Input_Text"].str.strip() != ""]
    df = df[df["Output"].str.strip() != ""]
    return df


def split_row_ids(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    row_ids = df["row_id"].to_numpy()
    return (
        row_ids[train_idx].tolist(),
        row_ids[val_idx].tolist(),
        row_ids[test_idx].tolist(),
    )


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def generate_splits(args):
    dataset_path = Path(args.dataset)
    df = load_and_clean_df(dataset_path)
    dataset_hash = sha256_file(dataset_path)

    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for split_id in range(args.num_splits):
        seed = args.base_seed + split_id
        train_ids, val_ids, test_ids = split_row_ids(
            df,
            seed=seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        manifest = SplitManifest(
            split_id=split_id,
            seed=seed,
            dataset_path=str(dataset_path),
            dataset_sha256=dataset_hash,
            total_rows=len(df),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            train_row_ids=train_ids,
            val_row_ids=val_ids,
            test_row_ids=test_ids,
        )

        manifest_path = splits_dir / f"split_{split_id:02d}.json"
        write_json(manifest_path, asdict(manifest))

        index_rows.append({
            "split_id": split_id,
            "seed": seed,
            "manifest_path": str(manifest_path),
            "train_size": len(train_ids),
            "val_size": len(val_ids),
            "test_size": len(test_ids),
        })

    write_jsonl(splits_dir / "splits_index.jsonl", index_rows)
    print(f"Wrote {len(index_rows)} split manifests to {splits_dir}")


def read_split_manifest(args) -> SplitManifest:
    if args.split_manifest:
        manifest_path = Path(args.split_manifest)
    else:
        index_path = Path(args.splits_index)
        if not index_path.exists():
            raise FileNotFoundError(f"Split index not found: {index_path}")
        rows = []
        with index_path.open("r") as f:
            for line in f:
                rows.append(json.loads(line))
        match = next((r for r in rows if r["split_id"] == args.split_id), None)
        if match is None:
            raise ValueError(f"split_id {args.split_id} not found in {index_path}")
        manifest_path = Path(match["manifest_path"])

    with manifest_path.open("r") as f:
        payload = json.load(f)
    return SplitManifest(**payload)


def make_split_dfs(df: pd.DataFrame, manifest: SplitManifest):
    df_by_id = df.set_index("row_id")
    train_df = df_by_id.loc[manifest.train_row_ids].reset_index()
    val_df = df_by_id.loc[manifest.val_row_ids].reset_index()
    test_df = df_by_id.loc[manifest.test_row_ids].reset_index()
    return train_df, val_df, test_df


def build_prompts(df: pd.DataFrame, tokenizer, system_prompt: str):
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        user_prompt = INSTRUCTION_TEMPLATE.format(input_text=row.Input_Text).strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = len(tokenizer(full_prompt, add_special_tokens=False)["input_ids"])
        rows.append({
            "sample_id": i,
            "row_id": int(row.row_id),
            "input_text": row.Input_Text,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "full_prompt": full_prompt,
            "prompt_tokens": prompt_tokens,
        })
    return rows


def safe_git_info():
    info = {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        info["git_commit"] = commit
        info["git_dirty"] = bool(status)
    except Exception:
        pass
    return info


def run_split(args):
    manifest = read_split_manifest(args)
    dataset_path = Path(manifest.dataset_path)

    df = load_and_clean_df(dataset_path)
    train_df, val_df, test_df = make_split_dfs(df, manifest)

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{RUN_NAME}_split{manifest.split_id:02d}_{run_stamp}"
    runs_root = Path(args.runs_root)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    split_manifest_path = run_dir / "split_manifest.json"
    write_json(split_manifest_path, asdict(manifest))

    train_df[["row_id", "Input_Text", "Output"]].to_csv(
        run_dir / "train_split.csv", index=False
    )
    val_df[["row_id", "Input_Text", "Output"]].to_csv(
        run_dir / "val_split.csv", index=False
    )
    test_df[["row_id", "Input_Text", "Output"]].to_csv(
        run_dir / "test_split.csv", index=False
    )

    set_seed(manifest.seed)

    model, tokenizer = setup_model_and_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    model = setup_peft_model(model)

    system_prompt = format_system_prompt()
    preprocess_function = create_preprocess_function(tokenizer, system_prompt)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=list(train_dataset.features),
    )
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=list(val_dataset.features),
    )
    tokenized_test = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=list(test_dataset.features),
    )

    output_dir = run_dir / "model"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        num_train_epochs=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="tensorboard",
        seed=manifest.seed,
        data_seed=manifest.seed,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        raw_eval_dataset=val_dataset,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_wrapper(eval_pred, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        report_dir=str(run_dir),
    )

    start_time = time.time()
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    trainer.raw_eval_dataset = test_dataset
    metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")

    epoch_label = int(trainer.state.epoch) if trainer.state.epoch is not None else "final"

    prompts = build_prompts(test_df, tokenizer, system_prompt)
    write_jsonl(run_dir / "test_prompts.jsonl", prompts)

    predictions_path = run_dir / f"predictions_epoch_{epoch_label}.csv"
    if predictions_path.exists():
        preds_df = pd.read_csv(predictions_path)
        if len(preds_df) == len(test_df):
            preds_df["row_id"] = test_df["row_id"].to_numpy()
            preds_df["sample_id"] = np.arange(len(test_df))
        preds_df.to_csv(run_dir / f"predictions_epoch_{epoch_label}_with_row_id.csv", index=False)

        if len(preds_df) == len(prompts):
            merged = []
            for i, prompt in enumerate(prompts):
                merged.append({
                    **prompt,
                    "ground_truth": preds_df.loc[i, "Ground_Truth"],
                    "prediction": preds_df.loc[i, "Predicted_Text"],
                })
            write_jsonl(run_dir / "test_predictions_with_prompts.jsonl", merged)

    meta = {
        "run_id": run_id,
        "split_id": manifest.split_id,
        "seed": manifest.seed,
        "dataset_path": manifest.dataset_path,
        "dataset_sha256": manifest.dataset_sha256,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "max_length": MAX_LENGTH,
        "base_model": BASE_MODEL_NAME,
        "system_prompt": system_prompt,
        "output_dir": str(output_dir),
        "report_dir": str(run_dir),
        "metrics": metrics,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "duration_seconds": round(time.time() - start_time, 2),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
    }
    meta.update(safe_git_info())

    write_json(run_dir / "run_meta.json", meta)

    print(f"Run complete: {run_id}")


def build_parser():
    parser = argparse.ArgumentParser(description="Split + finetune runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-splits", help="Create split manifests")
    gen.add_argument("--dataset", default=DATASET_PATH)
    gen.add_argument("--num-splits", type=int, default=30)
    gen.add_argument("--base-seed", type=int, default=1000)
    gen.add_argument("--train-ratio", type=float, default=0.8)
    gen.add_argument("--val-ratio", type=float, default=0.1)
    gen.add_argument("--test-ratio", type=float, default=0.1)
    gen.add_argument("--splits-dir", default="splits/dataset7_80_10_10")

    run = subparsers.add_parser("run", help="Run finetuning for a split")
    run.add_argument("--split-manifest", default=None)
    run.add_argument("--split-id", type=int, default=None)
    run.add_argument("--splits-index", default="splits/dataset7_80_10_10/splits_index.jsonl")
    run.add_argument("--runs-root", default="runs")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-splits":
        generate_splits(args)
        return

    if args.command == "run":
        if args.split_manifest is None and args.split_id is None:
            raise ValueError("Provide --split-manifest or --split-id")
        run_split(args)
        return


if __name__ == "__main__":
    main()
