from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: str | Path) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def load_model(path: str | Path) -> Any:
    return joblib.load(Path(path))
