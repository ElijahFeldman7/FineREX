import pandas as pd
import re
import os
from collections import defaultdict

def norm(text):
    return str(text).strip().lower() if pd.notna(text) else ""

def extract_tuples_from_cell(cell_value):
    if not isinstance(cell_value, str) or not cell_value.strip():
        return set()    
    pattern = r"\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)"
    matches = re.findall(pattern, cell_value)
    return {(norm(m[0]), norm(m[1])) for m in matches}

def parse_gt_output(gt_content):
    t_del = r'\{tuple_delimiter\}'
    ent_matches = re.findall(rf'\("entity"{t_del}(.*?){t_del}(.*?){t_del}', gt_content, re.I)
    ent_gt_set = {(norm(m[0]), norm(m[1])) for m in ent_matches}
    rel_matches = re.findall(rf'\("relationship"{t_del}(.*?){t_del}(.*?){t_del}', gt_content, re.I)
    rels_gt_set = {(norm(m[0]), norm(m[1])) for m in rel_matches}
    return ent_gt_set, rels_gt_set

def calc_stats(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

def run_pipeline(base_path='datasets/'):
    df_metrics = pd.read_csv(os.path.join(base_path, 'detailed_pairs.csv'))
    df_ds5 = pd.read_csv(os.path.join(base_path, 'dataset5.csv'))
    df_ds7 = pd.read_csv(os.path.join(base_path, 'dataset7.csv'))

    df_metrics = df_metrics.merge(df_ds5[['Sr.No.', 'Input_Text']], left_on='Row_ID', right_on='Sr.No.', how='inner')
    df_metrics['join_key'] = df_metrics['Input_Text'].apply(norm)

    agg_preds = defaultdict(lambda: {'ent': set(), 'rel': set()})
    for _, row in df_metrics.iterrows():
        key = row['join_key']
        agg_preds[key]['ent'] |= extract_tuples_from_cell(row['TP_entity_pairs'])
        agg_preds[key]['ent'] |= extract_tuples_from_cell(row['FP_entity_pairs'])
        agg_preds[key]['rel'] |= extract_tuples_from_cell(row['TP_relation_pairs'])
        agg_preds[key]['rel'] |= extract_tuples_from_cell(row['FP_relation_pairs'])

    g_ent, g_rel = {"tp": 0, "fp": 0, "fn": 0}, {"tp": 0, "fp": 0, "fn": 0}
    type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    final_rows = []

    for _, row in df_ds7.iterrows():
        input_txt = row['Input_Text']
        key = norm(input_txt)
        p_ent, p_rel = agg_preds[key]['ent'], agg_preds[key]['rel']
        gt_ent, gt_rel = parse_gt_output(row['Output'])
        
        tp_e, fp_e, fn_e = p_ent & gt_ent, p_ent - gt_ent, gt_ent - p_ent
        tp_r, fp_r, fn_r = p_rel & gt_rel, p_rel - gt_rel, gt_rel - p_rel

        g_ent['tp'] += len(tp_e); g_ent['fp'] += len(fp_e); g_ent['fn'] += len(fn_e)
        g_rel['tp'] += len(tp_r); g_rel['fp'] += len(fp_r); g_rel['fn'] += len(fn_r)

        for _, etype in tp_e: type_stats[etype.upper()]["tp"] += 1
        for _, etype in fn_e: type_stats[etype.upper()]["fn"] += 1
        for _, etype in fp_e: type_stats[etype.upper()]["fp"] += 1

        final_rows.append({
            'Sr.No.': row.get('Sr.No.', ''), 
            'Input_Text': input_txt,
            'TP_ent': len(tp_e), 'FP_ent': len(fp_e), 'FN_ent': len(fn_e),
            'TP_rel': len(tp_r), 'FP_rel': len(fp_r), 'FN_rel': len(fn_r),
            'TP_ent_pairs': "; ".join([str(x) for x in sorted(tp_e)]),
            'FP_ent_pairs': "; ".join([str(x) for x in sorted(fp_e)]),
            'FN_ent_pairs': "; ".join([str(x) for x in sorted(fn_e)]),
            'TP_rel_pairs': "; ".join([str(x) for x in sorted(tp_r)]),
            'FP_rel_pairs': "; ".join([str(x) for x in sorted(fp_r)]),
            'FN_rel_pairs': "; ".join([str(x) for x in sorted(fn_r)])
        })

    pd.DataFrame(final_rows).to_csv(os.path.join(base_path, 'recalculated_dataset7_metrics.csv'), index=False)

    with open(os.path.join(base_path, 'summary_report.txt'), 'w') as f:
        f.write("===== GLOBAL SUMMARY =====\n")
        for label, counts in [("Entities", g_ent), ("Relations", g_rel)]:
            p, r, f1 = calc_stats(counts['tp'], counts['fp'], counts['fn'])
            f.write(f"{label}: TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}\n")
            f.write(f"P={p:.4f}, R={r:.4f}, F1={f1:.4f}\n\n")
        f.write("===== PER-TYPE ENTITY SUMMARY =====\n")
        for etype, c in sorted(type_stats.items()):
            p, r, f1 = calc_stats(c['tp'], c['fp'], c['fn'])
            f.write(f"{etype}: TP={c['tp']}, FP={c['fp']}, FN={c['fn']} | P={p:.4f}, R={r:.4f}, F1={f1:.4f}\n")

if __name__ == "__main__":
    run_pipeline()