import networkx as nx
import matplotlib as plt
import numpy as np
from pathlib import Path
import json

INPUT_PATH = Path("/Users/eli/research/link-kg/datasets/processed_kg")

#01USVsJaquez
for path in INPUT_PATH.iterdir():
    subpath = Path(path)
    #entity
    for entitypath in subpath.iterdir():
        #json
        with open(entitypath / "final_memory.json", "r") as f:
            f