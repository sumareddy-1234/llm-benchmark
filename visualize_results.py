<<<<<<< HEAD
import os
import json
import matplotlib.pyplot as plt
import argparse

# -----------------------------
# CLI Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Visualize LLM Benchmark Results")
parser.add_argument("--results_dir", type=str, default="benchmark_results", help="Directory with JSON results")
args = parser.parse_args()

results_dir = args.results_dir

# -----------------------------
# Load all results
# -----------------------------
model_results = {}
for fname in os.listdir(results_dir):
    if fname.endswith(".json"):
        model_name = fname.replace("results_", "").replace(".json", "")
        with open(os.path.join(results_dir, fname), "r", encoding="utf-8") as f:
            model_results[model_name] = json.load(f)

# -----------------------------
# Prepare metrics
# -----------------------------
avg_latency = {}
avg_throughput = {}
ram_usages = {}
gpu_usages = {}

for model, entries in model_results.items():
    latencies = [e["latency_sec"] for e in entries]
    throughputs = [e["throughput_tokens_per_sec"] for e in entries]
    rams = [e["ram_usage_mb"] for e in entries]
    gpus = [e["gpu_usage_mb"] for e in entries if e.get("gpu_usage_mb") is not None]

    avg_latency[model] = sum(latencies) / len(latencies)
    avg_throughput[model] = sum(throughputs) / len(throughputs)
    ram_usages[model] = rams
    if gpus:
        gpu_usages[model] = gpus

# -----------------------------
# Plot Average Latency
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(avg_latency.keys(), avg_latency.values(), color='skyblue')
plt.ylabel("Average Latency (sec)")
plt.title("Average Latency per Model")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "avg_latency.png"))
plt.close()

# -----------------------------
# Plot RAM Usage Boxplot
# -----------------------------
plt.figure(figsize=(8,5))
plt.boxplot(ram_usages.values(), tick_labels=ram_usages.keys())
plt.ylabel("RAM Usage (MB)")
plt.title("RAM Usage per Model")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "ram_usage_boxplot.png"))
plt.close()

# -----------------------------
# Plot GPU Usage Boxplot (optional)
# -----------------------------
if gpu_usages:
    plt.figure(figsize=(8,5))
    plt.boxplot(gpu_usages.values(), tick_labels=gpu_usages.keys())
    plt.ylabel("GPU Usage (MB)")
    plt.title("GPU Usage per Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gpu_usage_boxplot.png"))
    plt.close()

print(f"✅ Visualization completed! Images saved in {results_dir}")
=======
import os
import json
import matplotlib.pyplot as plt
import argparse

# -----------------------------
# CLI Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Visualize LLM Benchmark Results")
parser.add_argument("--results_dir", type=str, default="benchmark_results", help="Directory with JSON results")
args = parser.parse_args()

results_dir = args.results_dir

# -----------------------------
# Load all results
# -----------------------------
model_results = {}
for fname in os.listdir(results_dir):
    if fname.endswith(".json"):
        model_name = fname.replace("results_", "").replace(".json", "")
        with open(os.path.join(results_dir, fname), "r", encoding="utf-8") as f:
            model_results[model_name] = json.load(f)

# -----------------------------
# Prepare metrics
# -----------------------------
avg_latency = {}
avg_throughput = {}
ram_usages = {}
gpu_usages = {}

for model, entries in model_results.items():
    latencies = [e["latency_sec"] for e in entries]
    throughputs = [e["throughput_tokens_per_sec"] for e in entries]
    rams = [e["ram_usage_mb"] for e in entries]
    gpus = [e["gpu_usage_mb"] for e in entries if e.get("gpu_usage_mb") is not None]

    avg_latency[model] = sum(latencies) / len(latencies)
    avg_throughput[model] = sum(throughputs) / len(throughputs)
    ram_usages[model] = rams
    if gpus:
        gpu_usages[model] = gpus

# -----------------------------
# Plot Average Latency
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(avg_latency.keys(), avg_latency.values(), color='skyblue')
plt.ylabel("Average Latency (sec)")
plt.title("Average Latency per Model")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "avg_latency.png"))
plt.close()

# -----------------------------
# Plot RAM Usage Boxplot
# -----------------------------
plt.figure(figsize=(8,5))
plt.boxplot(ram_usages.values(), tick_labels=ram_usages.keys())
plt.ylabel("RAM Usage (MB)")
plt.title("RAM Usage per Model")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "ram_usage_boxplot.png"))
plt.close()

# -----------------------------
# Plot GPU Usage Boxplot (optional)
# -----------------------------
if gpu_usages:
    plt.figure(figsize=(8,5))
    plt.boxplot(gpu_usages.values(), tick_labels=gpu_usages.keys())
    plt.ylabel("GPU Usage (MB)")
    plt.title("GPU Usage per Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gpu_usage_boxplot.png"))
    plt.close()

print(f"✅ Visualization completed! Images saved in {results_dir}")
>>>>>>> 7f0d7c36e623c1541999983b012b91d9714f2538
