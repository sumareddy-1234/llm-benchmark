from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .config import BenchmarkConfig
from .models import ModelWrapper
from .metrics import measure_generation, metrics_to_row


def run_benchmark(cfg: BenchmarkConfig, models: Dict[str, ModelWrapper]) -> pd.DataFrame:
    # load prompts
    df = pd.read_csv(cfg.dataset_path)
    prompts = df["prompt"].tolist()

    if cfg.max_prompts is not None:
        prompts = prompts[: cfg.max_prompts]

    rows: List[dict] = []

    total_runs = len(prompts) * len(models)
    progress = tqdm(total=total_runs, desc="Benchmarking", unit="run")

    for i, prompt in enumerate(prompts):
        for model_name, wrapper in models.items():
            metrics = measure_generation(wrapper, prompt, prompt_index=i)
            rows.append(metrics_to_row(metrics))
            progress.update(1)

    progress.close()

    results_df = pd.DataFrame(rows)
    return results_df
