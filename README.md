# LLM Benchmarking and Performance Analysis Toolkit

This project is a Python-based command-line application designed to evaluate and compare the performance of multiple Hugging Face transformer models using customizable text prompt datasets. The toolkit provides detailed benchmarking insights by measuring inference speed, throughput efficiency, and memory utilization, while also generating visual performance reports for analysis.

## Project Overview

The benchmarking pipeline automates the evaluation of different language models under a unified testing environment. It enables developers and researchers to analyze how model size impacts execution speed, computational cost, and resource consumption during text generation tasks.

The system exports structured CSV reports and visualization charts to simplify comparative analysis across models.

---

## Environment Setup

### 1. Create and Activate Virtual Environment

For Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Dataset and Benchmark Settings

* Prepare a CSV dataset containing a `prompt` column.
* Create or modify the benchmark configuration file in JSON format.

---

## Running the Application

Execute the benchmarking workflow using:

```bash
python -m benchmark_cli.cli --config config.example.json
```

---

## Generated Outputs

After execution, the system generates the following outputs inside the `outputs/` directory:

* `raw_metrics.csv`
  Detailed benchmarking results for each model and prompt combination, including latency, throughput, memory usage, and output statistics.

* `latency_comparison.png`
  Visualization comparing average inference latency across models.

* `memory_comparison.png`
  Visualization showing peak RAM utilization for each model.

---

## Supported Models

The default configuration evaluates multiple GPT-2 based transformer models with varying parameter sizes:

* `sshleifer/tiny-gpt2`
  Lightweight model optimized for faster inference and minimal memory usage.

* `distilgpt2`
  Compressed transformer model balancing performance and efficiency.

* `gpt2`
  Larger language model providing higher-quality outputs at increased computational cost.

---

## Performance Metrics Collected

The benchmarking framework records several important evaluation metrics:

* **Inference Latency**
  Total response generation time per prompt.

* **Token Throughput**
  Number of generated tokens processed per second.

* **Peak Memory Consumption**
  Maximum RAM utilized during inference execution.

* **Generated Output Length**
  Character-level measurement of generated responses.

---

## Benchmark Insights

Sample benchmark execution on CPU demonstrates clear trade-offs between model complexity and runtime efficiency:

* Smaller models achieve faster inference speeds with lower memory consumption.
* Larger transformer models provide more sophisticated text generation capabilities but require higher computational resources.
* Medium-sized distilled models offer a balanced compromise between performance quality and efficiency.

These observations help developers select appropriate language models based on deployment constraints, hardware availability, and application requirements.

---

## Technologies Used

* Python
* Hugging Face Transformers
* PyTorch
* Pandas
* Matplotlib
* psutil

---

## Use Cases

This project can be used for:

* Comparing transformer model efficiency
* Evaluating deployment feasibility on limited hardware
* Performance testing for NLP pipelines
* Educational demonstrations of LLM benchmarking concepts
* Research and experimentation with inference optimization
---
## Authour
Satti Suma
