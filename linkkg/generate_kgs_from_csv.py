#!/usr/bin/env python3
"""
Generate KGs directly from extraction.csv files.

Hybrid mode:
- Uses extraction.csv as source of truth for entities and relationships
- Optionally uses filtered coreference mappings only to consolidate aliases
"""

import os
import sys
import json
import csv
import re
import argparse
from collections import defaultdict
DEFAULT_PROCESSED_KG_DIR = "/Users/eli/research/link-kg/datasets/processed_kg"

try:
    import networkx as nx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: Please install networkx and matplotlib")
    sys.exit(1)


def load_extraction_csv(csv_path):
    """Load entities and relationships from extraction.csv"""
    entities_by_type = defaultdict(set)
    relationships = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                output_text = row.get('output_text', '').strip()
                
                # Parse both entity and relationship tuples
                for line in output_text.split('\n'):
                    line = line.strip()
                    
                    if line.startswith('("entity"'):
                        # Parse entity: ("entity"|NAME|TYPE|DESCRIPTION)
                        content = line[1:-1]  # Remove first and last chars ()
                        parts = content.split('|')
                        
                        if len(parts) >= 4:
                            entity_name = parts[1].strip()
                            entity_type = parts[2].strip().lower()
                            description = '|'.join(parts[3:]).strip()
                            
                            if entity_name and entity_type:
                                entities_by_type[entity_type].add((entity_name, description))
                    
                    elif line.startswith('("relationship"'):
                        # Parse relationship: ("relationship"|ENTITY1|ENTITY2|DESCRIPTION|CONFIDENCE)
                        content = line[1:-1]  # Remove first and last chars ()
                        parts = content.split('|')
                        
                        if len(parts) >= 5:
                            entity1 = parts[1].strip()
                            entity2 = parts[2].strip()
                            rel_desc = '|'.join(parts[3:-1]).strip()  # Join all parts except last
                            confidence = int(parts[-1].strip()) if parts[-1].strip().isdigit() else 5
                            
                            if entity1 and entity2:
                                relationships.append({
                                    'source': entity1,
                                    'target': entity2,
                                    'description': rel_desc,
                                    'confidence': confidence
                                })
    except Exception as e:
        print(f"Error reading CSV: {e}")
    
    return dict(entities_by_type), relationships


def normalize_text(text):
    """Normalize text for robust entity matching."""
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize(text):
    """Tokenize normalized text into simple alphanumeric tokens."""
    return [tok for tok in re.findall(r"[a-z0-9]+", normalize_text(text)) if len(tok) > 1]


def looks_like_alias(left, right):
    """Heuristic check for alias-like pairs to reduce hallucinated mappings."""
    nl = normalize_text(left)
    nr = normalize_text(right)

    if not nl or not nr:
        return False
    if nl == nr:
        return True
    if nl in nr or nr in nl:
        return True

    lt = set(tokenize(left))
    rt = set(tokenize(right))
    if not lt or not rt:
        return False

    overlap = len(lt & rt)
    min_size = min(len(lt), len(rt))
    return overlap >= 1 and (overlap / max(1, min_size)) >= 0.6


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_filtered_coref_aliases(case_name, extracted_entities, processed_kg_dir=DEFAULT_PROCESSED_KG_DIR):
    """
    Load coreference mappings and keep only safe alias-like mappings.

    Rules:
    - At least one side must match extraction entities
    - Pair must look alias-like
    - Only used for canonicalization, never to add new nodes
    """
    processed_case_dir = os.path.join(processed_kg_dir, case_name)

    extracted_norm_to_original = {
        normalize_text(name): name for name in extracted_entities if name
    }
    extracted_norms = set(extracted_norm_to_original.keys())

    stats = {
        "total_mappings": 0,
        "null_mappings": 0,
        "dropped_outside_extraction": 0,
        "dropped_non_alias": 0,
        "kept_mappings": 0,
        "merged_extracted_pairs": 0,
    }

    if not os.path.isdir(processed_case_dir):
        return {}, stats

    uf = UnionFind(extracted_norms)
    external_to_extracted = {}

    entity_type_dirs = [
        d for d in os.listdir(processed_case_dir)
        if os.path.isdir(os.path.join(processed_case_dir, d))
    ]

    for entity_type in entity_type_dirs:
        final_memory_path = os.path.join(processed_case_dir, entity_type, "final_memory.json")
        if not os.path.exists(final_memory_path):
            continue

        try:
            with open(final_memory_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
        except Exception:
            continue

        mappings = memory_data.get("RESOLVED_ENTITIES", {})
        for left, right in mappings.items():
            stats["total_mappings"] += 1

            if right is None:
                stats["null_mappings"] += 1
                continue

            nl = normalize_text(left)
            nr = normalize_text(right)
            if not nl or not nr:
                stats["dropped_non_alias"] += 1
                continue

            in_left = nl in extracted_norms
            in_right = nr in extracted_norms
            if not in_left and not in_right:
                stats["dropped_outside_extraction"] += 1
                continue

            if not looks_like_alias(left, right):
                stats["dropped_non_alias"] += 1
                continue

            stats["kept_mappings"] += 1

            # If both sides are known extracted entities, merge them.
            if in_left and in_right:
                uf.union(nl, nr)
                stats["merged_extracted_pairs"] += 1
            elif in_left:
                external_to_extracted[nr] = nl
            elif in_right:
                external_to_extracted[nl] = nr

    # Build canonical mapping for extracted entities.
    groups = defaultdict(list)
    for norm_name in extracted_norms:
        groups[uf.find(norm_name)].append(norm_name)

    alias_map = {}
    for group in groups.values():
        canonical_norm = max(
            group,
            key=lambda n: (len(extracted_norm_to_original.get(n, n)), extracted_norm_to_original.get(n, n))
        )
        canonical_original = extracted_norm_to_original.get(canonical_norm, canonical_norm)
        for member in group:
            alias_map[member] = canonical_original

    # Map external alias keys into extracted canonical names.
    for external_norm, extracted_norm in external_to_extracted.items():
        alias_map[external_norm] = alias_map.get(
            extracted_norm,
            extracted_norm_to_original.get(extracted_norm, extracted_norm)
        )

    return alias_map, stats


def merge_description(existing, incoming):
    """Merge node descriptions while keeping output compact."""
    existing = (existing or "").strip()
    incoming = (incoming or "").strip()

    if not existing:
        return incoming
    if not incoming:
        return existing
    if incoming in existing:
        return existing
    if existing in incoming:
        return incoming
    return f"{existing} | {incoming}"


def create_kg_from_extraction(case_name, csv_path, processed_kg_dir=DEFAULT_PROCESSED_KG_DIR):
    """Create a knowledge graph from extraction.csv"""

    entities_data, relationships = load_extraction_csv(csv_path)

    extracted_entities = set()
    for entities in entities_data.values():
        for entity_name, _ in entities:
            if entity_name:
                extracted_entities.add(entity_name)

    alias_map, alias_stats = load_filtered_coref_aliases(case_name, extracted_entities, processed_kg_dir)

    def canonicalize(name):
        return alias_map.get(normalize_text(name), name)

    G = nx.Graph()
    G.graph['name'] = case_name
    G.graph['alias_filter_stats'] = json.dumps(alias_stats)
    
    # Entity type colors
    color_map = {
        'person': '#FF6B6B',
        'location': '#4ECDC4',
        'organization': '#FFE66D',
        'routes': '#95E1D3',
        'means of transportation': '#C7CEEA',
        'means of communication': '#F7B7A3',
        'smuggled items': '#B5EAD7',
        'route': '#95E1D3',
        'transportation': '#C7CEEA',
        'means_of_transportation': '#C7CEEA',
        'means_of_communication': '#F7B7A3',
        'smuggled_items': '#B5EAD7',
    }
    
    # Add nodes
    for entity_type, entities in entities_data.items():
        normalized_type = entity_type.lower().replace('_', ' ')
        for entity_name, description in entities:
            if entity_name:
                canonical_name = canonicalize(entity_name)

                if canonical_name in G.nodes:
                    # Keep first node type and merge description/aliases.
                    G.nodes[canonical_name]['description'] = merge_description(
                        G.nodes[canonical_name].get('description', ''),
                        description
                    )
                    aliases = set(filter(None, G.nodes[canonical_name].get('aliases', '').split('|')))
                    aliases.add(entity_name)
                    G.nodes[canonical_name]['aliases'] = '|'.join(sorted(aliases))
                    continue

                G.add_node(
                    canonical_name,
                    node_type=normalized_type,
                    description=description,
                    aliases=entity_name
                )
    
    # Add edges from relationships
    for rel in relationships:
        source = canonicalize(rel['source'])
        target = canonicalize(rel['target'])
        
        # Only add edge if both entities exist in the graph
        if source in G.nodes() and target in G.nodes():
            if G.has_edge(source, target):
                # Aggregate repeated extraction relationships after canonicalization.
                existing_desc = G[source][target].get('description', '')
                G[source][target]['description'] = merge_description(existing_desc, rel['description'])
                G[source][target]['confidence'] = max(
                    int(G[source][target].get('confidence', 0)),
                    int(rel.get('confidence', 0))
                )
                G[source][target]['count'] = int(G[source][target].get('count', 1)) + 1
            else:
                G.add_edge(
                    source,
                    target,
                    relation='related_to',
                    description=rel['description'],
                    confidence=rel['confidence'],
                    count=1
                )

    return G, entities_data, alias_stats


def visualize_kg(G, output_path, case_name):
    """Visualize and save the knowledge graph."""
    try:
        fig, ax = plt.subplots(figsize=(16, 12))
        
        color_map = {
            'person': '#FF6B6B',
            'location': '#4ECDC4',
            'organization': '#FFE66D',
            'route': '#95E1D3',
            'routes': '#95E1D3',
            'means of transportation': '#C7CEEA',
            'transportation': '#C7CEEA',
            'means of communication': '#F7B7A3',
            'means_of_communication': '#F7B7A3',
            'smuggled items': '#B5EAD7',
            'smuggled_items': '#B5EAD7',
        }
        
        node_colors = [
            color_map.get(G.nodes[node].get('node_type', 'unknown'), '#CCCCCC')
            for node in G.nodes()
        ]
        
        if len(G) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = {}
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        
        ax.set_title(f"Knowledge Graph: {case_name} (from extraction.csv)", fontsize=16, fontweight='bold')
        
        legend_elements = [
            plt.scatter([], [], c=color, s=100, label=entity_type)
            for entity_type, color in color_map.items()
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"  ⚠ Failed to visualize: {e}")
        return False


def save_kg_stats(G, output_path, entities_data, alias_stats=None):
    """Save graph statistics."""
    try:
        stats = {
            "case": G.graph.get('name', 'unknown'),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G) if G.number_of_nodes() > 0 else 0,
            "is_connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
            "entities_by_type": {k: len(v) for k, v in entities_data.items()}
        }

        if alias_stats is not None:
            stats["coref_alias_filter"] = alias_stats
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        return True
    except Exception as e:
        print(f"  ⚠ Failed to save stats: {e}")
        return False


def save_kg_graphml(G, output_path):
    """Save knowledge graph in GraphML format."""
    try:
        # Create a copy with all attributes as strings (GraphML compatibility)
        G_export = nx.Graph()
        G_export.graph['name'] = G.graph.get('name', 'unknown')
        
        # Add nodes with string attributes
        for node, attrs in G.nodes(data=True):
            node_attrs = {}
            for key, value in attrs.items():
                node_attrs[key] = str(value)
            G_export.add_node(node, **node_attrs)
        
        # Add edges with string attributes
        for source, target, attrs in G.edges(data=True):
            edge_attrs = {}
            for key, value in attrs.items():
                edge_attrs[key] = str(value)
            G_export.add_edge(source, target, **edge_attrs)
        
        nx.write_graphml(G_export, output_path)
        return True
    except Exception as e:
        print(f"  ⚠ Failed to save GraphML: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate KGs from extraction.csv with optional coref alias mappings"
    )
    parser.add_argument(
        "--runs-dir",
        default="/Users/eli/research/link-kg/runs/eval_llama_finetuned/2026-04-25_19-50-47",
        help="Directory containing per-case extraction.csv files",
    )
    parser.add_argument(
        "--processed-kg-dir",
        default="/Users/eli/research/link-kg/datasets/processed_kg",
        help="Directory containing per-case final_memory.json mappings",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/eli/research/link-kg/datasets/kgs",
        help="Output directory for KGs",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case names to process (default: all cases in runs-dir)",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    processed_kg_dir = args.processed_kg_dir
    output_base_dir = args.output_dir

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted([
            d for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d))
        ])

    print(f"Generating KGs from extraction.csv for {len(cases)} cases...")

    for case_name in cases:
        csv_path = os.path.join(runs_dir, case_name, "extraction.csv")
        
        if not os.path.exists(csv_path):
            print(f"  ⊘ {case_name}: extraction.csv not found")
            continue
        
        print(f"  → {case_name}...", end=" ", flush=True)
        
        case_output_dir = os.path.join(output_base_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)
        
        try:
            # Create KG
            G, entities_data, alias_stats = create_kg_from_extraction(
                case_name,
                csv_path,
                processed_kg_dir,
            )
            
            if G.number_of_nodes() == 0:
                print("⊘ (no entities found)")
                continue
            
            # Save visualizations and data
            viz_path = os.path.join(case_output_dir, f"{case_name}_kg.png")
            stats_path = os.path.join(case_output_dir, "kg_stats.json")
            graphml_path = os.path.join(case_output_dir, f"{case_name}_kg.graphml")
            
            visualize_kg(G, viz_path, case_name)
            save_kg_stats(G, stats_path, entities_data, alias_stats=alias_stats)
            save_kg_graphml(G, graphml_path)

            print(
                f"✓ ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
                f"kept_alias={alias_stats.get('kept_mappings', 0)})"
            )
            
        except Exception as e:
            print(f"✗ {e}")
    
    print(f"\nKnowledge graphs saved to: {output_base_dir}")


if __name__ == "__main__":
    main()
