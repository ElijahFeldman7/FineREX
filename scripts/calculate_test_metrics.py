import pandas as pd
import re
from collections import defaultdict
from pathlib import Path

def extract_entity_types_from_pairs(pairs_str):
    entity_types = defaultdict(int)
    if pd.isna(pairs_str) or pairs_str == '':
        return entity_types
    
    pattern = r"\('[^']*',\s*'([^']+)'\)"
    matches = re.findall(pattern, str(pairs_str))
    for match in matches:
        entity_types[match] += 1
    
    return entity_types

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def main():
    run_path = Path('/Users/eli/research/link-kg/runs/llama_finetune_2026-03-01_17-02-01')
    predictions_file = run_path / 'predictions_epoch_4.csv'
    metrics_file = Path('/Users/eli/research/link-kg/datasets/recalculated_dataset7_metrics.csv')
    
    predictions_df = pd.read_csv(predictions_file)
    test_texts = set(predictions_df['Input_Text'].values)
    print(f"Found {len(test_texts)} test samples")
    
    metrics_df = pd.read_csv(metrics_file)
    
    test_metrics = metrics_df[metrics_df['Input_Text'].isin(test_texts)].copy()
    
    total_tp_ent = test_metrics['TP_ent'].sum()
    total_fp_ent = test_metrics['FP_ent'].sum()
    total_fn_ent = test_metrics['FN_ent'].sum()
    
    total_tp_rel = test_metrics['TP_rel'].sum()
    total_fp_rel = test_metrics['FP_rel'].sum()
    total_fn_rel = test_metrics['FN_rel'].sum()
    
    p_ent, r_ent, f1_ent = calculate_metrics(total_tp_ent, total_fp_ent, total_fn_ent)
    p_rel, r_rel, f1_rel = calculate_metrics(total_tp_rel, total_fp_rel, total_fn_rel)
    
    per_type_entities = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    
    for idx, row in test_metrics.iterrows():
        tp_pairs = extract_entity_types_from_pairs(row['TP_ent_pairs'])
        for entity_type, count in tp_pairs.items():
            per_type_entities[entity_type]['TP'] += count
        
        fp_pairs = extract_entity_types_from_pairs(row['FP_ent_pairs'])
        for entity_type, count in fp_pairs.items():
            per_type_entities[entity_type]['FP'] += count
        
        fn_pairs = extract_entity_types_from_pairs(row['FN_ent_pairs'])
        for entity_type, count in fn_pairs.items():
            per_type_entities[entity_type]['FN'] += count
    
    report = []
    report.append("GLOBAL SUMMARY")
    report.append(f"Entities: TP={total_tp_ent}, FP={total_fp_ent}, FN={total_fn_ent}")
    report.append(f"P={p_ent:.4f}, R={r_ent:.4f}, F1={f1_ent:.4f}")
    report.append("")
    report.append(f"Relations: TP={total_tp_rel}, FP={total_fp_rel}, FN={total_fn_rel}")
    report.append(f"P={p_rel:.4f}, R={r_rel:.4f}, F1={f1_rel:.4f}")
    report.append("")
    report.append("Per-type entity summary")
    
    for entity_type in sorted(per_type_entities.keys()):
        stats = per_type_entities[entity_type]
        tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
        p, r, f1 = calculate_metrics(tp, fp, fn)
        
        entity_type_upper = entity_type.upper()
        report.append(f"{entity_type_upper}: TP={tp}, FP={fp}, FN={fn} | P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    
    report_text = '\n'.join(report)
    print("\n" + report_text)
    
    output_file = Path('/Users/eli/research/link-kg/datasets/test_metrics_summary.txt')
    with open(output_file, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to {output_file}")

if __name__ == '__main__':
    main()
