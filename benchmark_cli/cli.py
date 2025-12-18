import argparse
import sys
from pathlib import Path

from .config import load_config
from .models import load_models
from .runner import run_benchmark
from .plots import plot_latency_bar, plot_memory_bar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM performance benchmarking tool"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.example.json",
        help="Path to config JSON file",
    )
    args = parser.parse_args()

    try:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        cfg = load_config(str(cfg_path))

        if not cfg.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {cfg.dataset_path}")

        print("Loaded config:")
        print("  dataset:", cfg.dataset_path)
        print("  models:", cfg.models)
        print("  device:", cfg.device)

        models = load_models(cfg.models, cfg.device, cfg.max_new_tokens)

        results_df = run_benchmark(cfg, models)

        output_csv = Path("outputs/raw_metrics.csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved raw metrics to {output_csv}")

        summary = results_df.groupby("model_name")[["latency_sec", "tokens_per_sec", "peak_ram_mb"]].mean()
        print("\nAverage metrics per model:")
        print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

        latency_png = "outputs/latency_comparison.png"
        memory_png = "outputs/memory_comparison.png"
        plot_latency_bar(results_df, latency_png)
        plot_memory_bar(results_df, memory_png)
        print(f"\nSaved charts:\n  {latency_png}\n  {memory_png}")

    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
