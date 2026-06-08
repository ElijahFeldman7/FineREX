import os
import json
import re
import glob
import matplotlib.pyplot as plt
import numpy as np

FINETUNE_DIR = "runs/llama_finetune_2026-01-09_23-28-44"
BASELINE_DIR = "runs/llama_baseline_2026-01-06_18-30-26"

def calculate_average_from_jsonl(file_path):
    totals = {
        "entity_f1": 0.0,
        "rel_f1": 0.0,
        "mae": 0.0,
        "count": 0
    }
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                metrics = data.get("metrics", {})
                
                totals["entity_f1"] += metrics.get("entity_f1", 0)
                totals["rel_f1"] += metrics.get("relationship_f1", 0)
                totals["mae"] += metrics.get("relationship_score_mae", 0)
                totals["count"] += 1
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    if totals["count"] == 0:
        return None

    return {
        "entity_f1": totals["entity_f1"] / totals["count"],
        "rel_f1": totals["rel_f1"] / totals["count"],
        "mae": totals["mae"] / totals["count"]
    }

def get_finetune_data():
    data = []
    files = glob.glob(os.path.join(FINETUNE_DIR, "metrics_epoch_*.jsonl"))
    
    print(f"DEBUG: Found {len(files)} JSONL files in Finetune dir.")

    for f in files:
        filename = os.path.basename(f)
        match = re.search(r"epoch_(\d+)", filename)
        if match:
            epoch = int(match.group(1))
            avgs = calculate_average_from_jsonl(f)
            if avgs:
                avgs["epoch"] = epoch
                data.append(avgs)
    
    data.sort(key=lambda x: x["epoch"])
    return data

def get_baseline_data():
    """Parses the metrics.jsonl from the baseline directory."""
    f_path = os.path.join(BASELINE_DIR, "metrics.jsonl")
    if not os.path.exists(f_path):
        return {"entity_f1": 0.0, "rel_f1": 0.0}
    
    avgs = calculate_average_from_jsonl(f_path)
    if not avgs:
        return {"entity_f1": 0.0, "rel_f1": 0.0}
    return avgs


def plot_learning_curve(data):
    epochs = [d["epoch"] for d in data]
    ent_f1 = [d["entity_f1"] for d in data]
    rel_f1 = [d["rel_f1"] for d in data]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ent_f1, marker='o', linewidth=2, label='Entity Extraction F1')
    plt.plot(epochs, rel_f1, marker='s', linewidth=2, label='Relationship Extraction F1')
    
    plt.title('Learning Curve: Extraction Performance over Epochs', fontsize=14)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('F1 Score (0.0 - 1.0)', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.savefig("graph_1_learning_curve.png", dpi=300)
    print("Saved graph_1_learning_curve.png")
    plt.close()

def plot_mae_curve(data):
    epochs = [d["epoch"] for d in data]
    mae = [d["mae"] for d in data]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mae, marker='^', color='purple', linewidth=2, label='Relationship Score MAE')
    
    plt.title('Accuracy of Relationship Strength Scores (1-10)', fontsize=14)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Mean Absolute Error (Lower is Better)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.savefig("graph_4_mae_curve.png", dpi=300)
    print("Saved graph_4_mae_curve.png")
    plt.close()

def plot_baseline_comparison(ft_data, bl_data):
    if not ft_data: return
    best_ft = ft_data[-1] #final epoch

    labels = ['Entity F1', 'Relationship F1']
    baseline_scores = [bl_data["entity_f1"], bl_data["rel_f1"]]
    finetuned_scores = [best_ft["entity_f1"], best_ft["rel_f1"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, baseline_scores, width, label='Baseline', color='gray', alpha=0.7)
    bars2 = plt.bar(x + width/2, finetuned_scores, width, label='Finetuned', color='#1f77b4')

    plt.title('Baseline vs. Finetuned', fontsize=14)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(x, labels, fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.savefig("graph_3_baseline_vs_finetune.png", dpi=300)
    print("Saved graph_3_baseline_vs_finetune.png")
    plt.close()


if __name__ == "__main__":
    print(f"Reading from: {FINETUNE_DIR}")
    ft_data = get_finetune_data()
    bl_data = get_baseline_data()

    if not ft_data:
        print(f"Error: No metrics_epoch_*.jsonl files found in {FINETUNE_DIR}")
    else:
        print(f"Successfully processed {len(ft_data)} epoch files.")
        print(f"Baseline Data: {bl_data}")
        
        plot_learning_curve(ft_data)
        plot_mae_curve(ft_data)
        plot_baseline_comparison(ft_data, bl_data)
        print("donex")