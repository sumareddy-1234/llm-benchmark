from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def run_model(model_name, prompts, device="cpu"):
    """
    Run a causal language model on a list of prompts and print the output.
    """
    print(f"\nRunning benchmark for model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=50)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}\nGenerated: {text}\n")
