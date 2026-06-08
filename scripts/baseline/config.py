import os
from datetime import datetime

DATASET_PATH = "datasets/dataset7.csv"
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_NAME = f"llama_baseline_{timestamp}"
NEW_MODEL_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(NEW_MODEL_DIR, exist_ok=True)
LOG_FILE = os.path.join(NEW_MODEL_DIR, "training_log.log")
