from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class BenchmarkConfig:
    dataset_path: Path
    models: list[str]
    device: str
    max_new_tokens: int
    batch_size: int
    max_prompts: int | None = None


def load_config(path: str) -> BenchmarkConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return BenchmarkConfig(
        dataset_path=Path(raw["dataset_path"]),
        models=list(raw["models"]),
        device=raw.get("device", "cpu"),
        max_new_tokens=int(raw.get("max_new_tokens", 64)),
        batch_size=int(raw.get("batch_size", 1)),
        max_prompts=(
            int(raw["max_prompts"]) if raw.get("max_prompts") is not None else None
        ),
    )
