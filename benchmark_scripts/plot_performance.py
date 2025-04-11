import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def process_log_files(log_dir):
    raw_data = defaultdict(lambda: defaultdict(list))

    log_files = glob.glob(os.path.join(log_dir, "*_srv_*_cli_*_*.csv"))

    for log_file in log_files:
        filename = os.path.basename(log_file).replace(".csv", "")
        parts = filename.split('_')
        policy = parts[0]
        num_server_threads = int(parts[2])
        num_clients = int(parts[4])

        df = pd.read_csv(log_file)
        last_throughput = df['overall_throughput'].iloc[-1]
        last_latency = df['overall_latency'].iloc[-1]
        last_timestamp = df['timestamp'].iloc[-1]

        key = (policy, num_server_threads)
        raw_data[key][num_clients].append((last_throughput, last_latency, last_timestamp))

    processed_data = {}
    for key, client_dict in raw_data.items():
        data_points = []
        for num_clients, values in client_dict.items():
            throughputs, latencies, runtimes = zip(*values)
            total_throughput = sum(throughputs)
            avg_latency = sum(latencies) / len(latencies)
            avg_runtime = sum(runtimes) / len(runtimes)
            data_points.append((num_clients, total_throughput, avg_latency, avg_runtime))
        data_points.sort(key=lambda x: x[0])
        processed_data[key] = data_points

    return processed_data

def plot_overlay(processed_data):
    # Extract unique policies and thread counts
    policies = sorted({policy for (policy, _) in processed_data})
    server_thread_counts = sorted({threads for (_, threads) in processed_data})

    # Line style per policy
    policy_styles = ['-', '--', '-.', ':']
    policy_to_style = {policy: policy_styles[i % len(policy_styles)] for i, policy in enumerate(policies)}

    # Log2-based gradient color mapping for threads
    log_server_thread_counts = [np.log2(threads) for threads in server_thread_counts]
    norm = mcolors.Normalize(vmin=min(log_server_thread_counts), vmax=max(log_server_thread_counts))
    colormap = cm.get_cmap('viridis')
    thread_to_color = {threads: colormap(norm(np.log2(threads))) for threads in server_thread_counts}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Total Throughput", "Average Latency", "Runtime"]
    ylabels = ["Throughput (requests/sec)", "Latency (ms)", "Runtime (seconds)"]

    for idx, metric_index in enumerate([1, 2, 3]):  # 1: throughput, 2: latency, 3: runtime
        ax = axes[idx]
        for (policy, num_server_threads), data in sorted(processed_data.items()):
            num_clients = [x[0] for x in data]
            metric_values = [x[metric_index] for x in data]
            label = f'{policy} ({num_server_threads} threads)'
            ax.plot(
                num_clients, metric_values,
                marker='o',
                linestyle=policy_to_style[policy],
                color=thread_to_color[num_server_threads],
                label=label
            )
        
        ax.set_title(f"{titles[idx]} vs Number of Clients", fontsize=14)
        ax.set_xlabel("Number of Clients (log scale)", fontsize=12)
        ax.set_ylabel(ylabels[idx], fontsize=12)
        ax.set_xscale('log', basex=2)  # Set X-axis to logarithmic scale
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Customize ticks for better readability
        ax.tick_params(axis='x', labelsize=10)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if idx == 2:  # only show legend once to avoid clutter
            ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.show()

log_directory = "logs/"
processed_data = process_log_files(log_directory)
plot_overlay(processed_data)
