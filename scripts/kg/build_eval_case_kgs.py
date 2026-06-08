import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Please install required packages (networkx, matplotlib)."
    ) from exc


RUN_ROOT = Path("/Users/eli/research/link-kg/runs/eval_llama_finetuned/2026-04-25_19-50-47")
FORMATTED_ROOT = Path("/Users/eli/research/link-kg/runs/eval_llama_finetuned/formatted")
FALLBACK_MEMORY_PATH = FORMATTED_ROOT / "01USVsJaquez" / "final_memory.json"
OUTPUT_ROOT = RUN_ROOT / "consolidated_case_kgs"

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
    return value.replace("_", " ").replace("-", " ")


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        key = normalize_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(value.strip())
    return ordered


def parse_entities_and_relationships(text: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str, float]]]:
    entities: List[Tuple[str, str, str]] = []
    relationships: List[Tuple[str, str, str, float]] = []

    for name, entity_type, description in ENTITY_PATTERN.findall(text or ""):
        name_clean = clean_text(name)
        if not name_clean:
            continue
        entities.append((name_clean, normalize_type(entity_type), clean_text(description)))

    for src, dst, description, score in REL_PATTERN.findall(text or ""):
        src_clean = clean_text(src)
        dst_clean = clean_text(dst)
        if not src_clean or not dst_clean:
            continue
        relationships.append((src_clean, dst_clean, clean_text(description), float(score)))

    return entities, relationships


def load_memory(memory_path: Path) -> Dict:
    with memory_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_memory_path(case_name: str) -> Path:
    candidate = FORMATTED_ROOT / case_name / "final_memory.json"
    if candidate.exists():
        return candidate
    return FALLBACK_MEMORY_PATH


def canonicalize(raw_name: str, resolved_entities: Dict[str, Optional[str]]) -> str:
    key = normalize_key(raw_name)
    resolved = resolved_entities.get(key)
    if resolved is None:
        return clean_text(raw_name)
    resolved_clean = clean_text(resolved)
    return resolved_clean if resolved_clean else clean_text(raw_name)


def dominant_type(type_counts: Counter) -> str:
    if not type_counts:
        return "unknown"
    return type_counts.most_common(1)[0][0]


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
        "unknown": "#adb5bd",
    }
    return palette.get(entity_type, "#adb5bd")


def read_extractions(case_dir: Path) -> List[Dict[str, str]]:
    extraction_file = case_dir / "extraction.csv"
    rows: List[Dict[str, str]] = []
    with extraction_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def build_case_graph(case_name: str, case_dir: Path, memory_payload: Dict) -> Dict[str, object]:
    resolved_raw = memory_payload.get("RESOLVED_ENTITIES", {})
    auxiliary_descriptions = memory_payload.get("AUXILIARY_DESCRIPTIONS", {})

    resolved_entities: Dict[str, Optional[str]] = {
        normalize_key(key): (clean_text(value) if isinstance(value, str) else None)
        for key, value in resolved_raw.items()
    }

    entity_profiles: Dict[str, Dict[str, object]] = {}
    edge_profiles: Dict[Tuple[str, str], Dict[str, object]] = {}

    for row in read_extractions(case_dir):
        output_text = row.get("output_text", "")
        chunk_number = row.get("chunk_number", "")
        input_text = row.get("input_text", "")

        entities, relationships = parse_entities_and_relationships(output_text)

        for raw_name, raw_type, raw_description in entities:
            canonical_name = canonicalize(raw_name, resolved_entities)
            profile = entity_profiles.setdefault(
                canonical_name,
                {
                    "aliases": [],
                    "type_counts": Counter(),
                    "descriptions": [],
                    "chunk_numbers": set(),
                    "input_texts": [],
                },
            )
            profile["aliases"].append(raw_name)
            profile["type_counts"][raw_type] += 1
            profile["descriptions"].extend(split_sentences(raw_description))
            profile["chunk_numbers"].add(chunk_number)
            if input_text:
                profile["input_texts"].append(input_text)

        for raw_src, raw_dst, rel_description, rel_score in relationships:
            src = canonicalize(raw_src, resolved_entities)
            dst = canonicalize(raw_dst, resolved_entities)
            edge_profile = edge_profiles.setdefault(
                (src, dst),
                {
                    "descriptions": [],
                    "scores": [],
                    "chunk_numbers": set(),
                    "raw_pairs": [],
                },
            )
            edge_profile["descriptions"].extend(split_sentences(rel_description))
            edge_profile["scores"].append(rel_score)
            edge_profile["chunk_numbers"].add(chunk_number)
            edge_profile["raw_pairs"].append((raw_src, raw_dst))

            entity_profiles.setdefault(
                src,
                {
                    "aliases": [raw_src],
                    "type_counts": Counter(),
                    "descriptions": [],
                    "chunk_numbers": {chunk_number},
                    "input_texts": [],
                },
            )
            entity_profiles.setdefault(
                dst,
                {
                    "aliases": [raw_dst],
                    "type_counts": Counter(),
                    "descriptions": [],
                    "chunk_numbers": {chunk_number},
                    "input_texts": [],
                },
            )

    for canonical_name, profile in entity_profiles.items():
        aux = auxiliary_descriptions.get(canonical_name)
        if isinstance(aux, str) and aux.strip():
            profile["descriptions"].extend(split_sentences(aux))

    graph = nx.DiGraph()

    node_records = []
    for canonical_name, profile in sorted(entity_profiles.items(), key=lambda item: item[0].lower()):
        aliases = dedupe_preserve_order(profile["aliases"])
        descriptions = dedupe_preserve_order(profile["descriptions"])
        entity_type = dominant_type(profile["type_counts"])
        merged_description = " ".join(descriptions)

        graph.add_node(
            canonical_name,
            label=canonical_name,
            entity_type=entity_type,
            aliases=" | ".join(aliases),
            description=merged_description,
            evidence_count=len(profile["chunk_numbers"]),
        )

        node_records.append(
            {
                "id": canonical_name,
                "label": canonical_name,
                "entity_type": entity_type,
                "aliases": aliases,
                "description": merged_description,
                "evidence_count": len(profile["chunk_numbers"]),
            }
        )

    edge_records = []
    for (src, dst), profile in sorted(edge_profiles.items(), key=lambda item: (item[0][0].lower(), item[0][1].lower())):
        descriptions = dedupe_preserve_order(profile["descriptions"])
        merged_description = " || ".join(descriptions)
        scores = profile["scores"]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        graph.add_edge(
            src,
            dst,
            description=merged_description,
            relationship_descriptions=" || ".join(descriptions),
            confidence_avg=round(avg_score, 4),
            evidence_count=len(profile["chunk_numbers"]),
        )

        edge_records.append(
            {
                "source": src,
                "target": dst,
                "description": merged_description,
                "relationship_descriptions": descriptions,
                "confidence_avg": round(avg_score, 4),
                "evidence_count": len(profile["chunk_numbers"]),
            }
        )

    case_output_dir = OUTPUT_ROOT / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    with (case_output_dir / "nodes.json").open("w", encoding="utf-8") as f:
        json.dump(node_records, f, ensure_ascii=False, indent=2)

    with (case_output_dir / "edges.json").open("w", encoding="utf-8") as f:
        json.dump(edge_records, f, ensure_ascii=False, indent=2)

    nx.write_graphml(graph, case_output_dir / "kg.graphml")

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    fig_width = max(12, min(28, 2 + node_count * 0.55))
    fig_height = max(10, min(24, 2 + node_count * 0.45))

    plt.figure(figsize=(fig_width, fig_height))
    if node_count > 1:
        # stronger force-directed layout tuned by graph area and node count
        area = fig_width * fig_height
        k_value = max(0.08, 0.9 * (area ** 0.5) / max(1.0, node_count ** 0.5))
        pos = nx.spring_layout(graph, seed=42, k=k_value, iterations=600)
        # normalize positions to [0,1] range to make plotting consistent
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

    node_colors = [node_color(graph.nodes[node].get("entity_type", "unknown")) for node in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=max(600, 1400 - 10 * node_count),
        node_color=node_colors,
        alpha=0.92,
        linewidths=0.7,
        edgecolors="#1f1f1f",
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        alpha=0.35,
        arrows=True,
        arrowsize=11,
        width=0.8,
    )
    label_font_size = 8 if node_count <= 12 else 6 if node_count <= 24 else 5
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: node for node in graph.nodes()},
        font_size=label_font_size,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.2),
    )

    plt.title(f"Consolidated KG: {case_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(case_output_dir / "kg.png", dpi=240)
    plt.close()

    # Save computed positions for reproducibility and external layout tuning
    try:
        with (case_output_dir / "positions.json").open("w", encoding="utf-8") as pf:
            json.dump({n: [float(pos[n][0]), float(pos[n][1])] for n in pos}, pf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    summary = {
        "case": case_name,
        "memory_source": str(resolve_memory_path(case_name)),
        "nodes": node_count,
        "edges": edge_count,
        "output_dir": str(case_output_dir),
        "graphml": str(case_output_dir / "kg.graphml"),
        "figure": str(case_output_dir / "kg.png"),
        "nodes_json": str(case_output_dir / "nodes.json"),
        "edges_json": str(case_output_dir / "edges.json"),
    }

    with (case_output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    for case_dir in sorted(p for p in RUN_ROOT.iterdir() if p.is_dir() and (p / "extraction.csv").exists()):
        case_name = case_dir.name
        memory_path = resolve_memory_path(case_name)
        memory_payload = load_memory(memory_path)
        summaries.append(build_case_graph(case_name, case_dir, memory_payload))

    manifest = {
        "run_root": str(RUN_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "cases": summaries,
        "case_count": len(summaries),
    }

    with (OUTPUT_ROOT / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
