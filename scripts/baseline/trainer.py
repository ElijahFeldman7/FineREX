import pandas as pd
import json
import os
import sys
import time
import getpass
import socket
import torch
from tqdm import tqdm
from scripts.llama_finetune.metrics import compute_metrics
import scripts.baseline.config as baseline_config

class CustomBaselineTrainer:
    def __init__(self, model, tokenizer, args, eval_dataset=None, system_prompt=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.eval_dataset = eval_dataset
        self.system_prompt = system_prompt
        self.save_run_description()

    def log(self, metrics):
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    def save_run_description(self):
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        desc = []
        desc.append("Run Type: baseline")
        desc.append(f"Start Time: {time.ctime()}")
        
        try:
            desc.append(f"Model ID: {self.model.name_or_path}")
        except AttributeError:
            desc.append("Model ID: Not available")
            
        desc.append(f"User: {getpass.getuser()}")
        desc.append(f"Hostname: {socket.gethostname()}")
        desc.append(f"Run Command: {' '.join(sys.argv)}")
        
        if self.eval_dataset:
            desc.append(f"Evaluation Dataset Size: {len(self.eval_dataset)}")

        if self.system_prompt:
            desc.append("\n--- System Prompt ---\n")
            desc.append(self.system_prompt)

        desc.append("\n--- Training Arguments ---\n")
        desc.append(self.args.to_json_string())

        desc.append("\n--- Custom Config (from scripts.baseline.config) ---\n")
        try:
            for key, value in vars(baseline_config).items():
                if not key.startswith("__") and not isinstance(value, type(sys)):
                    desc.append(f"{key}: {value}")
        except Exception:
            pass

        desc_path = os.path.join(output_dir, "desc.txt")
        with open(desc_path, "w") as f:
            f.write("\n".join(desc))

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        output_dir = self.args.output_dir
        summary_report_path = os.path.join(output_dir, "summary_report.txt")
        metrics_jsonl_path = os.path.join(output_dir, "metrics.jsonl")
        predictions_csv_path = os.path.join(output_dir, "predictions.csv")
        detailed_metrics_path = os.path.join(output_dir, "detailed_metrics.csv")

        if os.path.exists(summary_report_path):
            os.remove(summary_report_path)
        if os.path.exists(metrics_jsonl_path):
            os.remove(metrics_jsonl_path)
        if os.path.exists(predictions_csv_path):
            os.remove(predictions_csv_path)
        if os.path.exists(detailed_metrics_path):
            os.remove(detailed_metrics_path)

        total_metrics = {
            "parsability_score": 0.0,
            "entity_f1": 0.0,
            "relationship_f1": 0.0,
            "relationship_score_mae": 0.0
        }
        num_samples = 0
        all_predictions_data = []
        all_detailed_metrics = []

        print(f"Generating responses and saving to {summary_report_path}...")

        with open(summary_report_path, "w") as summary_writer, open(metrics_jsonl_path, "w") as jsonl_writer:
            
            for i, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
                full_prompt = sample["text"]
                ground_truth = sample["output"]
                
                if "Input_text:" in full_prompt:
                    display_prompt = full_prompt.split("Input_text:")[-1].strip()
                    if "Output:" in display_prompt:
                        display_prompt = display_prompt.split("Output:")[0].strip()
                else:
                    display_prompt = full_prompt 

                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
                input_length = inputs.input_ids.shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=2048, 
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                generated_tokens = outputs[0][input_length:]
                prediction_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                metrics = compute_metrics([prediction_text], [ground_truth])

                summary_writer.write(f"Input Data: {display_prompt}\n")
                summary_writer.write(f"Prediction: {prediction_text}\n")
                summary_writer.write(f"Ground Truth: {ground_truth}\n")
                summary_writer.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
                summary_writer.write("-" * 20 + "\n")

                metric_data = {
                    "prompt": display_prompt,
                    "prediction": prediction_text,
                    "ground_truth": ground_truth,
                    "metrics": metrics,
                }
                jsonl_writer.write(json.dumps(metric_data) + "\n")

                all_predictions_data.append({
                    "Input_text": display_prompt,
                    "Ground_Truth": ground_truth,
                    "Predicted_Text": prediction_text,
                })

                all_detailed_metrics.append({
                    "Row_ID": i + 1,
                    "TP_entities": metrics.get("entity_tp", 0),
                    "FP_entities": metrics.get("entity_fp", 0),
                    "FN_entities": metrics.get("entity_fn", 0),
                    "TP_rel": metrics.get("relationship_tp", 0),
                    "FP_rel": metrics.get("relationship_fp", 0),
                    "FN_rel": metrics.get("relationship_fn", 0),
                    "TP_entity_pairs": metrics.get("tp_entity_pairs", ""),
                    "FP_entity_pairs": metrics.get("fp_entity_pairs", ""),
                    "FN_entity_pairs": metrics.get("fn_entity_pairs", ""),
                    "TP_relation_pairs": metrics.get("tp_relation_pairs", ""),
                    "FP_relation_pairs": metrics.get("fp_relation_pairs", ""),
                    "FN_relation_pairs": metrics.get("fn_relation_pairs", ""),
                })

                for key in total_metrics:
                    if key in metrics and isinstance(metrics[key], (int, float)):
                        total_metrics[key] += metrics[key]

                num_samples += 1
        
        if all_predictions_data:
            predictions_df = pd.DataFrame(all_predictions_data)
            predictions_df.to_csv(predictions_csv_path, index=False)
            
            detailed_df = pd.DataFrame(all_detailed_metrics)
            detailed_df.to_csv(detailed_metrics_path, index=False)

        avg_metrics = {f"eval_{key}": value / num_samples for key, value in total_metrics.items() if num_samples > 0}
        
        all_decoded_preds = [d["Predicted_Text"] for d in all_predictions_data]
        all_decoded_gts = [d["Ground_Truth"] for d in all_predictions_data]
        final_global_metrics = compute_metrics(all_decoded_preds, all_decoded_gts)
        
        self.log(avg_metrics)

        try:
            header = "===== GLOBAL SUMMARY =====\n"
            header += f"Entities: TP={final_global_metrics.get('entity_tp', 0)}, FP={final_global_metrics.get('entity_fp', 0)}, FN={final_global_metrics.get('entity_fn', 0)}\n"
            header += f"P={final_global_metrics.get('entity_precision', 0):.4f}, R={final_global_metrics.get('entity_recall', 0):.4f}, F1={final_global_metrics.get('entity_f1', 0):.4f}\n\n"
            
            header += f"Relations: TP={final_global_metrics.get('relationship_tp', 0)}, FP={final_global_metrics.get('relationship_fp', 0)}, FN={final_global_metrics.get('relationship_fn', 0)}\n"
            header += f"P={final_global_metrics.get('relationship_precision', 0):.4f}, R={final_global_metrics.get('relationship_recall', 0):.4f}, F1={final_global_metrics.get('relationship_f1', 0):.4f}\n\n"
            
            header += "===== PER-TYPE ENTITY SUMMARY =====\n\n"
            entity_types = ["PERSON", "LOCATION", "ORGANIZATION", "MEANS_OF_TRANSPORTATION", "MEANS_OF_COMMUNICATION", "ROUTES", "SMUGGLED_ITEMS"]
            for etype in entity_types:
                tp = final_global_metrics.get(f"entity_{etype}_tp", 0)
                fp = final_global_metrics.get(f"entity_{etype}_fp", 0)
                fn = final_global_metrics.get(f"entity_{etype}_fn", 0)
                p = final_global_metrics.get(f"entity_{etype}_precision", 0)
                r = final_global_metrics.get(f"entity_{etype}_recall", 0)
                f1 = final_global_metrics.get(f"entity_{etype}_f1", 0)
                
                header += f"{etype}:\n"
                header += f"  TP={tp}, FP={fp}, FN={fn}\n"
                header += f"  P={p:.4f}, R={r:.4f}, F1={f1:.4f}\n\n"

            header += "--- GLOBAL METRICS (AVERAGE) ---\n"
            header += json.dumps(avg_metrics, indent=2) + "\n"
            header += "=" * 30 + "\n\n"
            
            with open(summary_report_path, "r") as f:
                existing_content = f.read()
            
            with open(summary_report_path, "w") as f:
                f.write(header + existing_content)
        except Exception as e:
            print(f"Error writing summary report header: {e}")
            pass

        return avg_metrics