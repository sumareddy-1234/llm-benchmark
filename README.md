# LLM Performance Benchmarking CLI

This project is a Python command-line tool for benchmarking multiple Hugging Face language models on a dataset of text prompts. It measures latency, throughput, and memory usage, and generates CSV reports and PNG charts for comparison.

## Setup

1. Create and activate a virtual environment.
   **for venv:**
.\.venv\Scripts\activate 

2. Install dependencies:

pip install -r requirements.txt

3. Prepare a prompts dataset (CSV with a `prompt` column) and a config file (JSON).

## Running the benchmark

python -m benchmark_cli.cli --config config.example.json

Outputs:

- `outputs/raw_metrics.csv` – one row per (prompt, model) with latency, tokens/sec, RAM, and output length.
- `outputs/latency_comparison.png` – bar chart of average latency per model.
- `outputs/memory_comparison.png` – bar chart of average peak RAM per model.

## Models and metrics

The default config benchmarks three open-source GPT-2 style models of increasing size:

- `sshleifer/tiny-gpt2` (small, fastest, lowest RAM)
- `distilgpt2` (medium)
- `gpt2` (largest, slowest, highest RAM)

Metrics collected:

- End-to-end latency per prompt (seconds).
- Throughput (generated tokens per second).
- Peak process RAM usage during generation (MB).
- Output length in characters (simple quality proxy).

## Results summary (example)

On the sample dataset of 3 prompts on CPU, the tool reports:

- `sshleifer/tiny-gpt2` has the lowest average latency and highest tokens/sec.
- `gpt2` has the highest latency and memory usage but can produce richer outputs.
- `distilgpt2` sits between them in both speed and resource usage.

These results illustrate the trade-off between model size, speed, and resource consumption when selecting an LLM for deployment.
