<<<<<<< HEAD
import os
import time
import argparse
import yaml
import json
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# -----------------------------
# GPU monitoring setup
# -----------------------------
pynvml_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    pynvml_available = True
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # first GPU
except:
    print("No NVIDIA GPU found or pynvml not working. GPU metrics will be skipped.")

# -----------------------------
# CLI Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="LLM Benchmark Tool")
parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
args = parser.parse_args()

# -----------------------------
# Load config
# -----------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Make sure config["models"] is a list of dicts
if isinstance(config["models"], list) and isinstance(config["models"][0], str):
    # convert to dict format
    config["models"] = [{"name": m} for m in config["models"]]

models = [m["name"] for m in config["models"]]
prompts_path = config.get("prompts_path", "prompts.txt")
output_dir = config.get("output_dir", "benchmark_results")
max_new_tokens = config.get("generation_params", {}).get("max_new_tokens", 50)
device = config.get("device", "cpu")

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load prompts
# -----------------------------
if prompts_path.endswith(".txt"):
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
elif prompts_path.endswith(".jsonl"):
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
else:
    raise ValueError("Unsupported prompts file format")

# -----------------------------
# Benchmark loop
# -----------------------------
for model_name in models:
    print(f"\nRunning benchmark for model: {model_name}")

    # Detect model type: T5/Flan are encoder-decoder
    if "t5" in model_name.lower() or "flan" in model_name.lower():
        ModelClass = AutoModelForSeq2SeqLM
    else:
        ModelClass = AutoModelForCausalLM

    # Load model safely
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ModelClass.from_pretrained(model_name)
        model.to(device)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        continue

    # Handle missing pad token
    if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # RAM usage
        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / 1024 ** 2  # MB

        # GPU usage
        gpu_before = 0
        if pynvml_available:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_before = meminfo.used / 1024 ** 2  # MB

        # Generate
        start_time = time.time()
        try:
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"Failed to generate for prompt: {e}")
            continue
        end_time = time.time()

        # RAM & GPU after
        ram_after = process.memory_info().rss / 1024 ** 2
        gpu_after = 0
        if pynvml_available:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_after = meminfo.used / 1024 ** 2

        latency = end_time - start_time
        tokens_generated = outputs.shape[1]
        throughput = tokens_generated / latency

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Additional metrics
        words = generated_text.split()
        avg_length = len(words)
        vocab_diversity = len(set(words)) / max(1, len(words))

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "latency_sec": latency,
            "throughput_tokens_per_sec": throughput,
            "ram_usage_mb": ram_after - ram_before,
            "gpu_usage_mb": (gpu_after - gpu_before) if pynvml_available else None,
            "avg_length": avg_length,
            "vocab_diversity": vocab_diversity
        })

    # Save results
    result_file = os.path.join(output_dir, f"results_{model_name.replace('/', '_')}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")

print("\n✅ Benchmark completed for all models!")
=======
import os
import time
import argparse
import yaml
import json
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# -----------------------------
# GPU monitoring setup
# -----------------------------
pynvml_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    pynvml_available = True
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # first GPU
except:
    print("No NVIDIA GPU found or pynvml not working. GPU metrics will be skipped.")

# -----------------------------
# CLI Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="LLM Benchmark Tool")
parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
args = parser.parse_args()

# -----------------------------
# Load config
# -----------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Make sure config["models"] is a list of dicts
if isinstance(config["models"], list) and isinstance(config["models"][0], str):
    # convert to dict format
    config["models"] = [{"name": m} for m in config["models"]]

models = [m["name"] for m in config["models"]]
prompts_path = config.get("prompts_path", "prompts.txt")
output_dir = config.get("output_dir", "benchmark_results")
max_new_tokens = config.get("generation_params", {}).get("max_new_tokens", 50)
device = config.get("device", "cpu")

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load prompts
# -----------------------------
if prompts_path.endswith(".txt"):
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
elif prompts_path.endswith(".jsonl"):
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
else:
    raise ValueError("Unsupported prompts file format")

# -----------------------------
# Benchmark loop
# -----------------------------
for model_name in models:
    print(f"\nRunning benchmark for model: {model_name}")

    # Detect model type: T5/Flan are encoder-decoder
    if "t5" in model_name.lower() or "flan" in model_name.lower():
        ModelClass = AutoModelForSeq2SeqLM
    else:
        ModelClass = AutoModelForCausalLM

    # Load model safely
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ModelClass.from_pretrained(model_name)
        model.to(device)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        continue

    # Handle missing pad token
    if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # RAM usage
        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / 1024 ** 2  # MB

        # GPU usage
        gpu_before = 0
        if pynvml_available:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_before = meminfo.used / 1024 ** 2  # MB

        # Generate
        start_time = time.time()
        try:
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"Failed to generate for prompt: {e}")
            continue
        end_time = time.time()

        # RAM & GPU after
        ram_after = process.memory_info().rss / 1024 ** 2
        gpu_after = 0
        if pynvml_available:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_after = meminfo.used / 1024 ** 2

        latency = end_time - start_time
        tokens_generated = outputs.shape[1]
        throughput = tokens_generated / latency

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Additional metrics
        words = generated_text.split()
        avg_length = len(words)
        vocab_diversity = len(set(words)) / max(1, len(words))

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "latency_sec": latency,
            "throughput_tokens_per_sec": throughput,
            "ram_usage_mb": ram_after - ram_before,
            "gpu_usage_mb": (gpu_after - gpu_before) if pynvml_available else None,
            "avg_length": avg_length,
            "vocab_diversity": vocab_diversity
        })

    # Save results
    result_file = os.path.join(output_dir, f"results_{model_name.replace('/', '_')}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")

print("\n✅ Benchmark completed for all models!")
>>>>>>> 7f0d7c36e623c1541999983b012b91d9714f2538
