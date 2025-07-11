import os
import glob
import pandas as pd
import numpy as np

def load_stats_from_dir(directory):
    stats_file = os.path.join(directory, "stats.csv")
    if os.path.isfile(stats_file):
        try:
            df = pd.read_csv(stats_file)
            return {
                "Total_runtime (seconds)": df["total"].iloc[0],
                "Max System memory (MB)": df["max_system_mem"].iloc[0],
                "Max GPU Memory (MB)": df["max_gpu_mem"].iloc[0]
            }
        except Exception as e:
            print(f"Failed to read {stats_file}: {e}")
    return None

def summarize_metrics(metrics):
    df = pd.DataFrame(metrics)
    return df.mean(), df.std()

def collect_metrics_by_prefix(prefix):
    pattern = f"{prefix}*"
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    results = []

    for d in dirs:
        data = load_stats_from_dir(d)
        if data:
            results.append(data)

    if not results:
        print(f"No valid data found for {prefix}")
        return None

    mean, std = summarize_metrics(results)
    print(f"\n== {prefix} Summary ==")
    print(mean)

# Run the analysis
collect_metrics_by_prefix("PemsIndex")
collect_metrics_by_prefix("PemsGPUIndex")

