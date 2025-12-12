<<<<<<< HEAD
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Your input text
input_text = "Once upon a time,"

# Tokenize with attention mask
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Generate text
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    pad_token_id=tokenizer.eos_token_id  # explicitly set pad token
)

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:\n", generated_text)
=======
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Your input text
input_text = "Once upon a time,"

# Tokenize with attention mask
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Generate text
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    pad_token_id=tokenizer.eos_token_id  # explicitly set pad token
)

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:\n", generated_text)
>>>>>>> 7f0d7c36e623c1541999983b012b91d9714f2538
