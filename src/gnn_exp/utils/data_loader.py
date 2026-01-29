from pathlib import Path
import json

def load_json(file_path):
    path = Path(file_path).expanduser().resolve()
    print(f"Loading JSON data from: {path}")
    with open(path, "r") as f:
        return json.load(f)
