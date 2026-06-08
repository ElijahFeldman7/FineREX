import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Please install required packages (networkx, matplotlib)."
    ) from exc


METRICS_JSONL_PATH = Path(
    "/Users/eli/research/link-kg/runs/llama_finetune_2026-03-01_17-02-01/metrics_epoch_4.jsonl"
)
MEMORY_JSON_PATH = Path(
    "/Users/eli/research/link-kg/runs/llama_finetune_2026-03-01_17-02-01/"
    "prepared_from_predictions_epoch4/person/final_memory.json"
)
OUTPUT_DIR = Path(
    "/Users/eli/research/link-kg/runs/llama_finetune_2026-03-01_17-02-01/consolidated_kg"
)

ENTITY_PATTERN = re.compile(
    r'\(\s*"?entity"?\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\)',
    re.IGNORECASE | re.DOTALL,
)
REL_PATTERN = re.compile(
    r'\(\s*"?relationship"?\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\)',
    re.IGNORECASE | re.DOTALL,
)


def clean_text(value: str) -> str:
    return value.strip().strip('"').strip("'").strip()


def normalize_key(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_type(value: str) -> str:
    value = normalize_key(value)
    value = value.replace("_", " ").replace("-", " ")
    return value


def split_description_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for v in values:
        k = normalize_key(v)
        if not k or k in seen:
            continue
        seen.add(k)
        ordered.append(v.strip())
    return ordered


def parse_prediction(prediction_text: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str, float]]]:
    entities: List[Tuple[str, str, str]] = []
    relationships: List[Tuple[str, str, str, float]] = []

    for name, etype, desc in ENTITY_PATTERN.findall(prediction_text or ""):
        entity_name = clean_text(name)
        entity_type = normalize_type(etype)
        entity_desc = clean_text(desc)
        if entity_name:
            entities.append((entity_name, entity_type, entity_desc))

    for src, dst, desc, score in REL_PATTERN.findall(prediction_text or ""):
        rel_src = clean_text(src)
        rel_dst = clean_text(dst)
        rel_desc = clean_text(desc)
        rel_score = float(score)
        if rel_src and rel_dst:
            relationships.append((rel_src, rel_dst, rel_desc, rel_score))

    return entities, relationships


def canonicalize_entity(raw_name: str, resolved_map: Dict[str, Optional[str]]) -> str:
    key = normalize_key(raw_name)
    resolved = resolved_map.get(key)
    if resolved is None:
        return clean_text(raw_name)
    resolved_clean = clean_text(resolved)
    return resolved_clean if resolved_clean else clean_text(raw_name)


def dominant_type(type_counter: Counter) -> str:
    if not type_counter:
        return "unknown"
    return type_counter.most_common(1)[0][0]


def node_color(entity_type: str) -> str:
    palette = {
        "person": "#f4a261",
        "location": "#2a9d8f",
        "organization": "#264653",
        "means of transportation": "#e76f51",
        "means of communication": "#577590",
        "routes": "#8ab17d",
        "smuggled items": "#9d4edd",
        "document": "#6c757d",
    }
    return palette.get(entity_type, "#adb5bd")


def build_kg() -> Dict[str, int]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with MEMORY_JSON_PATH.open("r", encoding="utf-8") as f:
        memory_payload = json.load(f)

    resolved_entities_raw = memory_payload.get("RESOLVED_ENTITIES", {})
    aux_descriptions = memory_payload.get("AUXILIARY_DESCRIPTIONS", {})

    resolved_entities: Dict[str, Optional[str]] = {
        normalize_key(k): (clean_text(v) if isinstance(v, str) else None)
        for k, v in resolved_entities_raw.items()
    }

    entity_profiles: Dict[str, Dict] = {}
    edge_profiles: Dict[Tuple[str, str], Dict] = {}

    with METRICS_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            prediction = row.get("prediction", "")

            entities, relationships = parse_prediction(prediction)

            for raw_name, raw_type, raw_desc in entities:
                canon = canonicalize_entity(raw_name, resolved_entities)
                profile = entity_profiles.setdefault(
                    canon,
                    {
                        "aliases": [],
                        "type_counter": Counter(),
                        "description_sentences": [],
                        "sample_ids": set(),
                    },
                )
                profile["aliases"].append(raw_name)
                if raw_type:
                    profile["type_counter"][raw_type] += 1
                profile["description_sentences"].extend(split_description_sentences(raw_desc))
                profile["sample_ids"].add(sample_id)

            for raw_src, raw_dst, rel_desc, rel_score in relationships:
                canon_src = canonicalize_entity(raw_src, resolved_entities)
                canon_dst = canonicalize_entity(raw_dst, resolved_entities)

                edge_key = (canon_src, canon_dst)
                edge = edge_profiles.setdefault(
                    edge_key,
                    {
                        "descriptions": [],
                        "scores": [],
                        "sample_ids": set(),
                        "raw_pairs": [],
                    },
                )
                edge["descriptions"].extend(split_description_sentences(rel_desc))
                edge["scores"].append(rel_score)
                edge["sample_ids"].add(sample_id)
                edge["raw_pairs"].append((raw_src, raw_dst))

                # Ensure nodes referenced only by relationships still exist.
                entity_profiles.setdefault(
                    canon_src,
                    {
                        "aliases": [raw_src],
                        "type_counter": Counter(),
                        "description_sentences": [],
                        "sample_ids": {sample_id},
                    },
                )
                entity_profiles.setdefault(
                    canon_dst,
                    {
                        "aliases": [raw_dst],
                        "type_counter": Counter(),
                        "description_sentences": [],
                        "sample_ids": {sample_id},
                    },
                )

    # Add auxiliary descriptions from memory for entities with matching canonical names.
    for canon, profile in entity_profiles.items():
        aux = aux_descriptions.get(canon)
        if isinstance(aux, str) and aux.strip():
            profile["description_sentences"].extend(split_description_sentences(aux))

    graph = nx.DiGraph()

    consolidated_entities = []
    for canon, profile in entity_profiles.items():
        aliases = dedupe_preserve_order(profile["aliases"])
        sentence_list = dedupe_preserve_order(profile["description_sentences"])
        entity_type = dominant_type(profile["type_counter"]) if profile["type_counter"] else "unknown"

        long_description = " ".join(sentence_list)

        graph.add_node(
            canon,
            label=canon,
            entity_type=entity_type,
            aliases=" | ".join(aliases),
            description=long_description,
            evidence_count=len(profile["sample_ids"]),
        )

        consolidated_entities.append(
            {
                "id": canon,
                "label": canon,
                "entity_type": entity_type,
                "aliases": aliases,
                "description": long_description,
                "evidence_count": len(profile["sample_ids"]),
            }
        )

    consolidated_edges = []
    for (src, dst), edge in edge_profiles.items():
        desc_list = dedupe_preserve_order(edge["descriptions"])
        merged_desc = " || ".join(desc_list)
        avg_score = sum(edge["scores"]) / len(edge["scores"]) if edge["scores"] else 0.0

        graph.add_edge(
            src,
            dst,
            description=merged_desc,
            confidence_avg=round(avg_score, 4),
            evidence_count=len(edge["sample_ids"]),
        )

        consolidated_edges.append(
            {
                "source": src,
                "target": dst,
                "description": merged_desc,
                "confidence_avg": round(avg_score, 4),
                "evidence_count": len(edge["sample_ids"]),
            }
        )

    # Save consolidated artifacts.
    with (OUTPUT_DIR / "entities_consolidated.json").open("w", encoding="utf-8") as f:
        json.dump(consolidated_entities, f, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "edges_consolidated.json").open("w", encoding="utf-8") as f:
        json.dump(consolidated_edges, f, ensure_ascii=False, indent=2)

    nx.write_graphml(graph, OUTPUT_DIR / "kg_consolidated.graphml")

    # Visualize with NetworkX.
    plt.figure(figsize=(24, 16))
    # physics-inspired spring layout scaled to graph size
    node_count = graph.number_of_nodes()
    if node_count > 1:
        area = 24 * 16
        k_value = max(0.08, 0.9 * (area ** 0.5) / max(1.0, node_count ** 0.5))
        pos = nx.spring_layout(graph, k=k_value, seed=42, iterations=800)
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        spanx = maxx - minx if maxx != minx else 1.0
        spany = maxy - miny if maxy != miny else 1.0
        for n, (x, y) in pos.items():
            pos[n] = ((x - minx) / spanx, (y - miny) / spany)
    else:
        pos = {next(iter(graph.nodes())): (0.5, 0.5)} if node_count == 1 else {}

    node_colors = [node_color(graph.nodes[n].get("entity_type", "unknown")) for n in graph.nodes()]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=500,
        node_color=node_colors,
        alpha=0.9,
        linewidths=0.5,
        edgecolors="#1f1f1f",
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        alpha=0.35,
        arrows=True,
        arrowsize=10,
        width=0.7,
    )

    labels = {n: n for n in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7)

    plt.title("Consolidated Knowledge Graph (Canonicalized)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kg_consolidated_networkx.png", dpi=240)
    plt.close()

    # Save positions for reproducibility
    try:
        with (OUTPUT_DIR / "positions.json").open("w", encoding="utf-8") as pf:
            json.dump({n: [float(pos[n][0]), float(pos[n][1])] for n in pos}, pf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    summary = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "output_dir": str(OUTPUT_DIR),
        "graphml": str(OUTPUT_DIR / "kg_consolidated.graphml"),
        "figure": str(OUTPUT_DIR / "kg_consolidated_networkx.png"),
        "entities_json": str(OUTPUT_DIR / "entities_consolidated.json"),
        "edges_json": str(OUTPUT_DIR / "edges_consolidated.json"),
    }

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    summary = build_kg()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
