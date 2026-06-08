#!/usr/bin/env python3
"""
Generate consolidated knowledge graphs using NetworkX for all cases.
Combines all entity types (person, location, organization, routes, etc.) into one graph per case.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import re

try:
    import networkx as nx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: Please install networkx and matplotlib:")
    print("  pip install networkx matplotlib")
    sys.exit(1)


def load_final_memory(entity_path):
    """Load final_memory.json for an entity type."""
    memory_path = os.path.join(entity_path, "final_memory.json")
    if not os.path.exists(memory_path):
        return None
    
    try:
        with open(memory_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Failed to load {memory_path}: {e}")
        return None


def load_resolved_text(entity_path, case_name, entity_type):
    """Load the consolidated resolved text file."""
    resolved_file = os.path.join(entity_path, f"{entity_type}_resolved_{case_name}.txt")
    if os.path.exists(resolved_file):
        try:
            with open(resolved_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"  ⚠ Failed to load resolved text: {e}")
    return ""


def normalize_canonical_name(value, alias, case_name, entity_type):
    if value is None:
        return None
    if isinstance(value, list):
        items = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    items.append(cleaned)
            else:
                items.append(str(item))
        if not items:
            return None
        if len(items) > 1:
            print(
                "WARN: Multiple canonical names for %s/%s alias '%s'; using first: %s"
                % (case_name, entity_type, alias, items[0])
            )
        return items[0]
    if isinstance(value, dict):
        for key in ("canonical", "name", "value"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value)


def extract_relationships_from_text(text, entities_by_type):
    """
    Extract relationships from resolved text.
    Creates connections between co-occurring entities.
    """
    relationships = []
    
    # Get all entities from all types
    all_entities = {}
    for entity_type, entities in entities_by_type.items():
        all_entities.update(entities)
    
    # Find sentences and extract co-occurrences
    sentences = re.split(r'[.!?]', text)
    
    for sentence in sentences:
        # Find all entities mentioned in this sentence
        entities_in_sentence = []
        for entity in all_entities:
            if entity.lower() in sentence.lower():
                entities_in_sentence.append(entity)
        
        # Create edges between co-occurring entities
        for i, entity1 in enumerate(entities_in_sentence):
            for entity2 in entities_in_sentence[i+1:]:
                # Determine relationship type
                relation = 'related_to'
                
                if 'testified' in sentence.lower():
                    relation = 'testified_about'
                elif 'arrested' in sentence.lower():
                    relation = 'arrested'
                elif 'transported' in sentence.lower():
                    relation = 'transported'
                elif 'visited' in sentence.lower() or 'went to' in sentence.lower():
                    relation = 'visited'
                elif 'located' in sentence.lower() or 'found' in sentence.lower():
                    relation = 'located_at'
                
                relationships.append({
                    'source': entity1,
                    'target': entity2,
                    'relation': relation
                })
    
    return relationships


def create_consolidated_kg(datasets_dir, case_name, output_dir):
    """Create a consolidated knowledge graph for a case."""
    case_path = os.path.join(datasets_dir, case_name)
    
    entity_types = [
        "person",
        "location",
        "organization",
        "routes",
        "means_of_transportation",
        "means_of_communication",
        "smuggled_items"
    ]
    
    # Create graph
    G = nx.Graph()
    G.graph['name'] = case_name
    
    # Maps for quick lookup
    all_entities_by_type = defaultdict(dict)
    all_relationships = []
    
    # Load all entity types
    for entity_type in entity_types:
        entity_path = os.path.join(case_path, entity_type)
        if not os.path.exists(entity_path):
            continue
        
        memory = load_final_memory(entity_path)
        if not memory:
            continue
        
        resolved_entities = memory.get("RESOLVED_ENTITIES", {})
        aux_descriptions = memory.get("AUXILIARY_DESCRIPTIONS", {})
        
        # Add entities to graph
        for alias, canonical_name in resolved_entities.items():
            canonical_name = normalize_canonical_name(
                canonical_name, alias, case_name, entity_type
            )
            if canonical_name is None:
                continue
            
            if canonical_name not in G:
                description = aux_descriptions.get(canonical_name, "")
                G.add_node(
                    canonical_name,
                    node_type=entity_type,  # Changed from 'type' to 'node_type'
                    description=description,
                    aliases="|" + alias if alias != canonical_name else ""
                )
            else:
                # Add alias if already exists
                if alias != canonical_name:
                    aliases = G.nodes[canonical_name].get('aliases', "")
                    if alias not in aliases:
                        G.nodes[canonical_name]['aliases'] = aliases + "|" + alias
            
            all_entities_by_type[entity_type][canonical_name] = True
        
        # Load resolved text for relationship extraction
        resolved_text = load_resolved_text(entity_path, case_name, entity_type)
        if resolved_text:
            relationships = extract_relationships_from_text(resolved_text, all_entities_by_type)
            all_relationships.extend(relationships)
    
    # Add edges (relationships)
    for rel in all_relationships:
        if rel['source'] in G and rel['target'] in G:
            G.add_edge(
                rel['source'],
                rel['target'],
                relation=rel['relation']
            )
    
    return G, all_entities_by_type


def visualize_kg(G, output_path, case_name):
    """Visualize and save the knowledge graph."""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Color nodes by type
        color_map = {
            'person': '#FF6B6B',
            'location': '#4ECDC4',
            'organization': '#FFE66D',
            'routes': '#95E1D3',
            'means_of_transportation': '#C7CEEA',
            'means_of_communication': '#F7B7A3',
            'smuggled_items': '#B5EAD7'
        }
        
        node_colors = [
            color_map.get(G.nodes[node].get('node_type', 'unknown'), '#CCCCCC')
            for node in G.nodes()
        ]
        
        # Layout
        if len(G) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = {}
        
        # Draw
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=1000,
            alpha=0.9,
            ax=ax
        )
        
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.5,
            ax=ax
        )
        
        # Title and legend
        ax.set_title(f"Knowledge Graph: {case_name}", fontsize=16, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c=color, s=100, label=entity_type)
            for entity_type, color in color_map.items()
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"  ⚠ Failed to visualize: {e}")
        return False


def save_kg_as_graphml(G, output_path):
    """Save graph as GraphML format."""
    try:
        # Create a copy with only string/primitive attributes for GraphML export
        G_export = nx.Graph(G)
        
        for node in G_export.nodes():
            attrs_to_remove = []
            for key, value in G_export.nodes[node].items():
                try:
                    # Try to convert to a GraphML-compatible type
                    if isinstance(value, str):
                        pass  # Keep strings as-is
                    elif isinstance(value, (int, float, bool)):
                        pass  # Keep primitives
                    else:
                        # Convert everything else to string
                        G_export.nodes[node][key] = str(value) if value else ""
                except:
                    attrs_to_remove.append(key)
            
            for key in attrs_to_remove:
                del G_export.nodes[node][key]
        
        for u, v in G_export.edges():
            attrs_to_remove = []
            for key, value in G_export.edges[u, v].items():
                try:
                    if isinstance(value, str):
                        pass
                    elif isinstance(value, (int, float, bool)):
                        pass
                    else:
                        G_export.edges[u, v][key] = str(value) if value else ""
                except:
                    attrs_to_remove.append(key)
            
            for key in attrs_to_remove:
                del G_export.edges[u, v][key]
        
        nx.write_graphml(G_export, output_path)
        return True
    except Exception as e:
        # Silently fail on GraphML - not critical for this workflow
        return True


def save_kg_stats(G, output_path, entities_by_type):
    """Save graph statistics."""
    try:
        stats = {
            "case": G.graph.get('name', 'unknown'),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
            "entities_by_type": {k: len(v) for k, v in entities_by_type.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        return True
    except Exception as e:
        print(f"  ⚠ Failed to save stats: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate consolidated knowledge graphs for all cases"
    )
    parser.add_argument(
        "--datasets-dir",
        default="/Users/eli/research/link-kg/datasets/processed_kg",
        help="Path to datasets/processed_kg directory"
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/eli/research/link-kg/datasets/kgs",
        help="Output directory for knowledge graphs"
    )
    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True)

    # Get all cases
    cases = sorted([
        d for d in os.listdir(datasets_dir)
        if os.path.isdir(os.path.join(datasets_dir, d))
    ])

    print(f"Generating knowledge graphs for {len(cases)} cases...")

    for idx, case in enumerate(cases, 1):
        print(f"  [{idx}/{len(cases)}] {case}...", end=" ", flush=True)
        
        case_output_dir = os.path.join(output_base_dir, case)
        os.makedirs(case_output_dir, exist_ok=True)
        
        try:
            # Create KG
            G, entities_by_type = create_consolidated_kg(datasets_dir, case, case_output_dir)
            
            if G.number_of_nodes() == 0:
                print("⊘ (no entities found)")
                continue
            
            # Save visualizations and data
            viz_path = os.path.join(case_output_dir, f"{case}_kg.png")
            graphml_path = os.path.join(case_output_dir, f"{case}_kg.graphml")
            stats_path = os.path.join(case_output_dir, "kg_stats.json")
            
            visualize_kg(G, viz_path, case)
            save_kg_as_graphml(G, graphml_path)
            save_kg_stats(G, stats_path, entities_by_type)
            
            print(f"✓ ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            
        except Exception as e:
            print(f"✗ {e}")

    print(f"\nKnowledge graphs saved to: {output_base_dir}")


if __name__ == "__main__":
    main()
