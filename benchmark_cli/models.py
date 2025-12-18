from typing import Dict, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ModelWrapper:
    def __init__(self, name: str, device: str = "cpu", max_new_tokens: int = 64):
        self.name = name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).to(self.device)
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def generate_with_token_count(self, prompt: str) -> Tuple[str, int]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        # new tokens after the prompt
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        num_tokens = gen_ids.shape[0]
        return text, int(num_tokens)

    
    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        text, _ = self.generate_with_token_count(prompt)
        return text


def load_models(model_names, device: str, max_new_tokens: int) -> Dict[str, ModelWrapper]:
    models: Dict[str, ModelWrapper] = {}
    for name in model_names:
        print(f"Loading model: {name} on {device} ...")
        models[name] = ModelWrapper(name, device=device, max_new_tokens=max_new_tokens)
    return models
