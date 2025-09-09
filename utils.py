import json
from pathlib import Path

def save_json(path: Path, data: dict):
    """Guarda un dict en un archivo JSON"""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
