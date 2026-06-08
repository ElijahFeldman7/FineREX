import pandas as pd
import os
import json
import sys
import time
import re
import torch
from tqdm import tqdm
import getpass
import socket
from transformers import Trainer
from .metrics import compute_metrics
from . import config as llama_finetune_config

def normalize_extraction(text):
    if not isinstance(text, str):
        return text
    text=text.lower()
    text=re.sub(r'\$\s+', '$', text)
    text=re.sub(r'\s+', ' ', text)
    return text.strip()

class CustomTrainer(Trainer):
    def __init__(self, *args, raw_eval_dataset=None, system_prompt=None, report_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_eval_dataset = raw_eval_dataset
        self.system_prompt = system_prompt
        self.report_dir = report_dir if report_dir is not None else self.args.output_dir
        
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
            
        self.save_run_description()

    def save_run_description(self):
        desc = []
        desc.append("Run Type: llama_finetune")
        desc.append(f"Start Time: {time.ctime()}")

        try:
            desc.append(f"Model ID: {self.model.name_or_path}")
        except AttributeError:
            desc.append("Model ID: Not available")
            
        desc.append(f"User: {getpass.getuser()}")
        desc.append(f"Hostname: {socket.gethostname()}")
        desc.append(f"Run Command: {' '.join(sys.argv)}")

        if self.train_dataset:
            desc.append(f"Training Dataset Size: {len(self.train_dataset)}")
        if self.eval_dataset:
            desc.append(f"Evaluation Dataset Size: {len(self.eval_dataset)}")
        
        if self.system_prompt:
            desc.append("\n--- System Prompt ---\n")
            desc.append(self.system_prompt)
            
        desc.append("\n--- Training Arguments ---\n")
        desc.append(self.args.to_json_string())

        desc.append("\n--- Custom Config (from scripts.llama_finetune.config) ---\n")
        for key, value in vars(llama_finetune_config).items():
            if not key.startswith("__") and not isinstance(
                value, type(sys)
            ):
                desc.append(f"{key}: {value}")

        desc_path = os.path.join(self.report_dir, "desc.txt")
        with open(desc_path, "w") as f:
            f.write("\n".join(desc))

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
             raise ValueError("Trainer: No evaluation dataset found (eval_dataset is None).")
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if eval_output is None:
            eval_output = {}
        self.model.eval()

        tokenizer = getattr(self, "processing_class", self.tokenizer)
        
        original_texts = [item['Input_Text'] for item in self.raw_eval_dataset]
        original_ground_truths = [item['Output'] for item in self.raw_eval_dataset]

        decoded_predictions = []
        decoded_ground_truths = []
        INSTRUCTION_TEMPLATE = """Input_text: \n{input_text}\nOutput:\n"""
        for i in tqdm(range(len(original_texts)), desc="Evaluating Samples"):
            user_content=INSTRUCTION_TEMPLATE.format(input_text=original_texts[i]).strip()
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            prompt_length = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2056,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    use_cache=True
                )
            generated_ids=output_ids[0][prompt_length:]
            pred_text=tokenizer.decode(generated_ids, skip_special_tokens=True)
            if "<END>" in pred_text:
                pred_text = pred_text.split("<END>")[0] + "<END>"
            pred_text= pred_text.strip()
            gt = original_ground_truths[i].strip()
            gt = gt.replace("{tuple_delimiter}", "|").replace("{record_delimiter}", "\n").replace("{completion_delimiter}", "<END>")

            pred_text = normalize_extraction(pred_text)
            gt = normalize_extraction(gt)

            decoded_predictions.append(pred_text)
            decoded_ground_truths.append(gt)

        



        global_metrics = compute_metrics(decoded_predictions, decoded_ground_truths)
        
        for key, value in global_metrics.items():
            eval_output[f"{metric_key_prefix}_{key}"] = value

        summary_report_parts = []
        jsonl_report = []
        all_detailed_metrics = []

        for i, (pred, label, prompt_text) in enumerate(zip(decoded_predictions, decoded_ground_truths, original_texts)):
            
            sample_metrics = compute_metrics([pred], [label])

            report_part = f"--- Sample {i} ---\n"
            report_part += f"Input Data: {prompt_text}\n\n" 
            report_part += f"Prediction:\n{pred}\n"
            report_part += f"Ground Truth:\n{label}\n"
            report_part += f"Metrics: {json.dumps(sample_metrics, indent=2)}\n\n"
            
            summary_report_parts.append(report_part)

            jsonl_report.append({
                "sample_id": i,
                "prompt": prompt_text,
                "prediction": pred,
                "ground_truth": label,
                "metrics": sample_metrics
            })
            
            all_detailed_metrics.append({
                "Row_ID": i + 1,
                "TP_entities": sample_metrics.get("entity_tp", 0),
                "FP_entities": sample_metrics.get("entity_fp", 0),
                "FN_entities": sample_metrics.get("entity_fn", 0),
                "TP_rel": sample_metrics.get("relationship_tp", 0),
                "FP_rel": sample_metrics.get("relationship_fp", 0),
                "FN_rel": sample_metrics.get("relationship_fn", 0),
                "TP_entity_pairs": sample_metrics.get("tp_entity_pairs", ""),
                "FP_entity_pairs": sample_metrics.get("fp_entity_pairs", ""),
                "FN_entity_pairs": sample_metrics.get("fn_entity_pairs", ""),
                "TP_relation_pairs": sample_metrics.get("tp_relation_pairs", ""),
                "FP_relation_pairs": sample_metrics.get("fp_relation_pairs", ""),
                "FN_relation_pairs": sample_metrics.get("fn_relation_pairs", ""),
            })
        
        summary_report_header = "===== GLOBAL SUMMARY =====\n"
        summary_report_header += f"Entities: TP={global_metrics.get('entity_tp', 0)}, FP={global_metrics.get('entity_fp', 0)}, FN={global_metrics.get('entity_fn', 0)}\n"
        summary_report_header += f"P={global_metrics.get('entity_precision', 0):.4f}, R={global_metrics.get('entity_recall', 0):.4f}, F1={global_metrics.get('entity_f1', 0):.4f}\n\n"
        
        summary_report_header += f"Relations: TP={global_metrics.get('relationship_tp', 0)}, FP={global_metrics.get('relationship_fp', 0)}, FN={global_metrics.get('relationship_fn', 0)}\n"
        summary_report_header += f"P={global_metrics.get('relationship_precision', 0):.4f}, R={global_metrics.get('relationship_recall', 0):.4f}, F1={global_metrics.get('relationship_f1', 0):.4f}\n\n"
        
        summary_report_header += "===== PER-TYPE ENTITY SUMMARY =====\n\n"
        entity_types = ["PERSON", "LOCATION", "ORGANIZATION", "MEANS_OF_TRANSPORTATION", "MEANS_OF_COMMUNICATION", "ROUTES", "SMUGGLED_ITEMS"]
        for etype in entity_types:
            tp = global_metrics.get(f"entity_{etype}_tp", 0)
            fp = global_metrics.get(f"entity_{etype}_fp", 0)
            fn = global_metrics.get(f"entity_{etype}_fn", 0)
            p = global_metrics.get(f"entity_{etype}_precision", 0)
            r = global_metrics.get(f"entity_{etype}_recall", 0)
            f1 = global_metrics.get(f"entity_{etype}_f1", 0)
            
            summary_report_header += f"{etype}:\n"
            summary_report_header += f"  TP={tp}, FP={fp}, FN={fn}\n"
            summary_report_header += f"  P={p:.4f}, R={r:.4f}, F1={f1:.4f}\n\n"

        summary_report_header += "--- GLOBAL METRICS (AVERAGE) ---\n"
        summary_report_header += json.dumps(eval_output, indent=2) + "\n"
        summary_report_header += "=" * 30 + "\n\n"
        
        summary_report = summary_report_header + "".join(summary_report_parts)

        summary_path = os.path.join(self.report_dir, "summary_report.txt")
        with open(summary_path, "w") as f:
            f.write(summary_report)

        if self.state.epoch is not None:
            epoch_label = int(self.state.epoch) 
        else:
            epoch_label = "final"
            
        jsonl_filename = f"metrics_epoch_{epoch_label}.jsonl"
        jsonl_path = os.path.join(self.report_dir, jsonl_filename)
        
        with open(jsonl_path, "w") as f:
            for entry in jsonl_report:
                f.write(json.dumps(entry) + "\n")

        predictions_df = pd.DataFrame({
            "Input_Text": original_texts,
            "Ground_Truth": decoded_ground_truths,
            "Predicted_Text": decoded_predictions
        })
        predictions_csv_path = os.path.join(self.report_dir, f"predictions_epoch_{epoch_label}.csv")
        predictions_df.to_csv(predictions_csv_path, index=False)

        detailed_df = pd.DataFrame(all_detailed_metrics)
        detailed_metrics_path = os.path.join(self.report_dir, f"detailed_metrics_epoch_{epoch_label}.csv")
        detailed_df.to_csv(detailed_metrics_path, index=False)

        return eval_output