# benchmark/benchmark_runner.py

from .model_runner import run_model
from .data_loader import load_prompts

def run_benchmark(config):
    """
    Runs benchmark for all models defined in the config on the specified dataset.
    """
    # Ensure 'dataset' key exists
    if "dataset" not in config:
        raise KeyError("Config file must contain a 'dataset' key pointing to your dataset path.")
    
    dataset_path = config["dataset"]
    
    # If dataset_path is a dict (like from YAML), extract string path
    if isinstance(dataset_path, dict):
        # Assuming the YAML has { 'path': '...' } structure
        dataset_path = dataset_path.get("path")
        if not dataset_path:
            raise ValueError("Dataset path not found in config['dataset']")
    
    # Load prompts from the dataset file
    prompts = load_prompts(dataset_path)

    # Run benchmark for each model in the config
    if "models" not in config or not config["models"]:
        raise KeyError("Config file must contain a 'models' key with a list of models.")

    for model_cfg in config["models"]:
        name = model_cfg.get("name")
        if not name:
            raise ValueError("Each model config must have a 'name' key.")
        
        print(f"\nRunning benchmark for model: {name}")
        run_model(name, prompts, device=config.get("device", "cpu"))
