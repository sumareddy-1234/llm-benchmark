import time
from dataclasses import dataclass
from typing import Dict, Any

import psutil
import torch


@dataclass
class RunMetrics:
    model_name: str
    prompt_index: int
    prompt: str
    latency_sec: float
    tokens_generated: int
    tokens_per_sec: float
    peak_ram_mb: float
    output_length_chars: int


def measure_generation(model_wrapper, prompt: str, prompt_index: int) -> RunMetrics:
    """
    Run one prompt through one model and record timing + memory metrics.
    """
    process = psutil.Process()
    start_mem = process.memory_info().rss
    start_time = time.perf_counter()

    # generate text
    output_text, tokens_generated = model_wrapper.generate_with_token_count(prompt)

    end_time = time.perf_counter()
    end_mem = process.memory_info().rss

    latency = end_time - start_time
    tokens_per_sec = tokens_generated / latency if latency > 0 and tokens_generated > 0 else 0.0
    peak_ram_mb = (end_mem - start_mem) / (1024 * 1024)

    return RunMetrics(
        model_name=model_wrapper.name,
        prompt_index=prompt_index,
        prompt=prompt,
        latency_sec=latency,
        tokens_generated=int(tokens_generated),
        tokens_per_sec=float(tokens_per_sec),
        peak_ram_mb=float(peak_ram_mb),
        output_length_chars=len(output_text),
    )


def metrics_to_row(m: RunMetrics) -> Dict[str, Any]:
    return {
        "model_name": m.model_name,
        "prompt_index": m.prompt_index,
        "prompt": m.prompt,
        "latency_sec": m.latency_sec,
        "tokens_generated": m.tokens_generated,
        "tokens_per_sec": m.tokens_per_sec,
        "peak_ram_mb": m.peak_ram_mb,
        "output_length_chars": m.output_length_chars,
    }
