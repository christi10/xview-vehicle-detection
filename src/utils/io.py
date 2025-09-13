import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
