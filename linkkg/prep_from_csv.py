import os
import argparse
import pandas as pd
import json
import re

TYPE_MAPPING = {
    "person": "PERSON",
    "location": "LOCATION",
    "organization": "ORGANIZATION",
    "org": "ORGANIZATION",
    "routes": "ROUTES",
    "mot": "MEANS_OF_TRANSPORTATION",
    "means_of_transportation": "MEANS_OF_TRANSPORTATION",
    "moc": "MEANS_OF_COMMUNICATION",
    "means_of_communication": "MEANS_OF_COMMUNICATION",
    "smuggleditems": "SMUGGLED_ITEMS",
    "smuggled_items": "SMUGGLED_ITEMS"
}

def parse_entities(output_text, target_entity_type=None):
    if not isinstance(output_text, str):
        return {
            "ENTITIES": {"PROPER_NOUN": [], "NOUN_PHRASE": []},
            "PROPER_NOUN_DESCRIPTION": {}
        }
    pattern = r'\("entity"\|(.*?)\|(.*?)\|(.*?)\)'
    matches = re.findall(pattern, output_text, re.DOTALL)
    
    entities = {
        "PROPER_NOUN": [],
        "NOUN_PHRASE": []
    }
    descriptions = {}
    
    mapped_target = TYPE_MAPPING.get(target_entity_type.lower(), target_entity_type.upper()) if target_entity_type else None
    
    for name, etype, desc in matches:
        name = name.strip()
        etype = etype.strip().upper()
        desc = desc.strip()
        
        if mapped_target and etype != mapped_target:
            continue
            
        entities["PROPER_NOUN"].append(name)
        descriptions[name] = desc
        
    return {
        "ENTITIES": entities,
        "PROPER_NOUN_DESCRIPTION": descriptions
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--entity-type", help="Target entity type to filter (e.g., person, mot)")
    args = parser.parse_args()
    
    chunk_out_dir = os.path.join(args.output_dir, "chunk_outputs")
    ner_out_dir = os.path.join(args.output_dir, "ner_outputs")
    
    os.makedirs(chunk_out_dir, exist_ok=True)
    os.makedirs(ner_out_dir, exist_ok=True)
    
    df = pd.read_csv(args.csv_file)
    
    for _, row in df.iterrows():
        chunk_val = row.get('chunk_number')
        if pd.isna(chunk_val):
            continue
        try:
            chunk_num = int(chunk_val)
        except (TypeError, ValueError):
            continue

        input_text = row.get('input_text')
        output_text = row.get('output_text')
        if pd.isna(input_text):
            input_text = ""
        if pd.isna(output_text):
            output_text = ""
        if not isinstance(input_text, str):
            input_text = str(input_text)
        if not isinstance(output_text, str):
            output_text = str(output_text)
        
        chunk_filename = f"chunk_{chunk_num:02d}.txt"
        with open(os.path.join(chunk_out_dir, chunk_filename), 'w', encoding='utf-8') as f:
            f.write(input_text)
            
        ner_data = parse_entities(output_text, args.entity_type)
        ner_filename = f"chunk_{chunk_num:02d}.json"
        with open(os.path.join(ner_out_dir, ner_filename), 'w', encoding='utf-8') as f:
            json.dump(ner_data, f, indent=2)
            
    print(f"Prepared {len(df)} chunks in {args.output_dir} for entity type {args.entity_type}")

if __name__ == "__main__":
    main()
