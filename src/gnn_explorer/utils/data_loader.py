import json
import pathlib

def load_json(self, file_path: str, root=pathlib.Path(__file__).resolve().parents[2]) -> dict:
    """Load JSON graph data and push to widget."""
    full_path = root / file_path
    print(f"Loading JSON data from: {file_path}")
    with open(full_path, "r") as f:
        data = json.load(f)
    return data  # sync to TS
