import pandas as pd

def calculate_metrics(results):
    df = pd.DataFrame(results)
    df["throughput"] = df["tokens_generated"] / df["latency"]
    return df

def save_results(df, output_path):
    df.to_csv(output_path, index=False)
