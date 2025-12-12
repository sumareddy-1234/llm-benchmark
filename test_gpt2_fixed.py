<<<<<<< HEAD
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Input text
prompt = "Once upon a time, "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate text
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,       # enable random sampling
    top_k=50,             # limit to top 50 words
    top_p=0.95,           # nucleus sampling
    temperature=0.8       # randomness factor
)

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:\n", generated_text)
=======
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Input text
prompt = "Once upon a time, "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate text
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,       # enable random sampling
    top_k=50,             # limit to top 50 words
    top_p=0.95,           # nucleus sampling
    temperature=0.8       # randomness factor
)

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:\n", generated_text)
>>>>>>> 7f0d7c36e623c1541999983b012b91d9714f2538
