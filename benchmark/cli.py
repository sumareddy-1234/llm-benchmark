import yaml
from .benchmark_runner import run_benchmark

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_benchmark(config)

if __name__ == "__main__":
    main()
