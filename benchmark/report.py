import matplotlib.pyplot as plt

def plot_metrics(df, output_dir):
    df_mean = df.groupby("model").mean()
    df_mean[["latency", "cpu_memory", "gpu_memory"]].plot(kind="bar")
    plt.title("Average Performance Metrics")
    plt.savefig(f"{output_dir}/metrics.png")
