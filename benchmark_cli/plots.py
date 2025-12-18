from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_latency_bar(df: pd.DataFrame, output_path: str) -> None:
    grouped = df.groupby("model_name")["latency_sec"].mean().sort_values()
    models = grouped.index.tolist()
    values = grouped.values

    plt.figure(figsize=(8, 4))
    plt.bar(models, values, color="#4ade80")
    plt.ylabel("Average latency (s)")
    plt.title("Average latency per model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_memory_bar(df: pd.DataFrame, output_path: str) -> None:
    grouped = df.groupby("model_name")["peak_ram_mb"].mean().sort_values()
    models = grouped.index.tolist()
    values = grouped.values

    plt.figure(figsize=(8, 4))
    plt.bar(models, values, color="#60a5fa")
    plt.ylabel("Average peak RAM (MB)")
    plt.title("Average peak RAM per model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
