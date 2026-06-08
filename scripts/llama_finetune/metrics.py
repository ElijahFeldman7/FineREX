import re
import numpy as np
import torch
from functools import partial

tuple_delimiter = "|" 

def parse_for_eval(text: str) -> (set, dict, bool):
    entities = set()
    relationships = {} 

    text = text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
    text = text.strip()

    d = re.escape(tuple_delimiter)

    entity_pattern = re.compile(
        r'\(\s*"entity"\s*' + d + r'\s*(.*?)\s*' + d + r'\s*(.*?)\s*' + d + r'.*?\)', 
        re.DOTALL
    )
    
    rel_pattern_with_score = re.compile(
        r'\(\s*"relationship"\s*' + d + r'\s*(.*?)\s*' + d + r'\s*(.*?)\s*' + d + r'.*?' + d + r'\s*(\d+)\s*\)', 
        re.DOTALL
    )

    try:
        for name, type_ in entity_pattern.findall(text):
            entities.add((name.strip().upper(), type_.strip().upper()))
        
        for ent1, ent2, score in rel_pattern_with_score.findall(text):
            rel_tuple = (ent1.strip().upper(), ent2.strip().upper())
            relationships[rel_tuple] = int(score)

        has_content = len(entities) > 0 or len(relationships) > 0
        is_parsable = has_content
        
        return entities, relationships, is_parsable
    except Exception:
        return set(), {}, False

def compute_metrics(predictions: list, ground_truths: list):
    parsability_scores, score_errors = [], []
    entity_tp, entity_fp, entity_fn = 0, 0, 0
    rel_tp, rel_fp, rel_fn = 0, 0, 0
    
    entity_types = ["PERSON", "LOCATION", "ORGANIZATION", "MEANS_OF_TRANSPORTATION", "MEANS_OF_COMMUNICATION", "ROUTES", "SMUGGLED_ITEMS"]
    per_type_counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in entity_types}

    tp_entity_list, fp_entity_list, fn_entity_list = [], [], []
    tp_rel_list, fp_rel_list, fn_rel_list = [], [], []

    for pred_str, label_str in zip(predictions, ground_truths):
        pred_entities, pred_rels, is_parsable = parse_for_eval(pred_str)
        label_entities, label_rels, _ = parse_for_eval(label_str)

        parsability_scores.append(1 if is_parsable else 0)

        if not is_parsable:
            entity_fn += len(label_entities)
            rel_fn += len(label_rels)
            for _, etype in label_entities:
                if etype in per_type_counts:
                    per_type_counts[etype]["fn"] += 1
            
            fn_entity_list.append("; ".join([str(e) for e in sorted(label_entities)]))
            tp_entity_list.append("")
            fp_entity_list.append("")
            
            fn_rel_list.append("; ".join([str(r) for r in sorted(label_rels.keys())]))
            tp_rel_list.append("")
            fp_rel_list.append("")
            continue

        tp_entities = pred_entities & label_entities
        fp_entities = pred_entities - label_entities
        fn_entities = label_entities - pred_entities

        entity_tp += len(tp_entities)
        entity_fp += len(fp_entities)
        entity_fn += len(fn_entities)

        for _, etype in tp_entities:
            if etype in per_type_counts:
                per_type_counts[etype]["tp"] += 1
        for _, etype in fp_entities:
            if etype in per_type_counts:
                per_type_counts[etype]["fp"] += 1
        for _, etype in fn_entities:
            if etype in per_type_counts:
                per_type_counts[etype]["fn"] += 1

        pred_rel_keys = set(pred_rels.keys())
        label_rel_keys = set(label_rels.keys())
        
        tp_rels = pred_rel_keys & label_rel_keys
        fp_rels = pred_rel_keys - label_rel_keys
        fn_rels = label_rel_keys - pred_rel_keys

        rel_tp += len(tp_rels)
        rel_fp += len(fp_rels)
        rel_fn += len(fn_rels)
        
        for rel_pair in tp_rels:
            if rel_pair in pred_rels and rel_pair in label_rels:
                error = abs(pred_rels[rel_pair] - label_rels[rel_pair])
                score_errors.append(error)

        def format_entity_set(eset):
            return "; ".join([f"('{n.lower()}', '{t.lower()}')" for n, t in sorted(list(eset))])
        
        def format_rel_set(rset):
            return "; ".join([f"('{r[0].lower()}', '{r[1].lower()}')" for r in sorted(list(rset))])

        tp_entity_list.append(format_entity_set(tp_entities))
        fp_entity_list.append(format_entity_set(fp_entities))
        fn_entity_list.append(format_entity_set(fn_entities))
        
        tp_rel_list.append(format_rel_set(tp_rels))
        fp_rel_list.append(format_rel_set(fp_rels))
        fn_rel_list.append(format_rel_set(fn_rels))

    final_metrics = {}
    final_metrics["parsability_score"] = np.mean(parsability_scores) if parsability_scores else 0.0

    def calc_prec_rec_f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    e_prec, e_rec, e_f1 = calc_prec_rec_f1(entity_tp, entity_fp, entity_fn)
    final_metrics['entity_precision'] = e_prec
    final_metrics['entity_recall'] = e_rec
    final_metrics['entity_f1'] = e_f1
    final_metrics['entity_tp'] = entity_tp
    final_metrics['entity_fp'] = entity_fp
    final_metrics['entity_fn'] = entity_fn

    r_prec, r_rec, r_f1 = calc_prec_rec_f1(rel_tp, rel_fp, rel_fn)
    final_metrics['relationship_precision'] = r_prec
    final_metrics['relationship_recall'] = r_rec
    final_metrics['relationship_f1'] = r_f1
    final_metrics['relationship_tp'] = rel_tp
    final_metrics['relationship_fp'] = rel_fp
    final_metrics['relationship_fn'] = rel_fn

    final_metrics['relationship_score_mae'] = np.mean(score_errors) if score_errors else 0.0
    
    for etype, counts in per_type_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p, r, f1 = calc_prec_rec_f1(tp, fp, fn)
        final_metrics[f"entity_{etype}_tp"] = tp
        final_metrics[f"entity_{etype}_fp"] = fp
        final_metrics[f"entity_{etype}_fn"] = fn
        final_metrics[f"entity_{etype}_precision"] = p
        final_metrics[f"entity_{etype}_recall"] = r
        final_metrics[f"entity_{etype}_f1"] = f1

    if len(predictions) == 1:
        final_metrics["tp_entity_pairs"] = tp_entity_list[0]
        final_metrics["fp_entity_pairs"] = fp_entity_list[0]
        final_metrics["fn_entity_pairs"] = fn_entity_list[0]
        final_metrics["tp_relation_pairs"] = tp_rel_list[0]
        final_metrics["fp_relation_pairs"] = fp_rel_list[0]
        final_metrics["fn_relation_pairs"] = fn_rel_list[0]

    return final_metrics

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def compute_metrics_wrapper(eval_pred, tokenizer):
    predicted_ids, labels = eval_pred
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    predicted_strings = tokenizer.batch_decode(predicted_ids, skip_special_tokens=False)
    label_strings = tokenizer.batch_decode(labels, skip_special_tokens=True)

    header_token = "<|start_header_id|>assistant<|end_header_id|>"
    
    clean_preds = []
    clean_labels = []

    for pred, label in zip(predicted_strings, label_strings):
        if header_token in pred:
            pred_content = pred.split(header_token)[-1]
        else:
            pred_content = pred

        pred_content = pred_content.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
        
        label_content = label.split(header_token)[-1] if header_token in label else label
        
        clean_preds.append(pred_content)
        clean_labels.append(label_content.strip())
    
    return compute_metrics(clean_preds, clean_labels)