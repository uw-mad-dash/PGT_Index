import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

def extract_nodes(s):
    match = re.search(r'node_(\d+)', s)
    if match:
        return int(match.group(1))
    return None  # or raise an error


try:
    user_input = input("Enter a value in seconds for the GPU baseline: ")
    gpu_baseline = float(user_input)
    
except ValueError:
    print("Invalid input. Please enter a numeric value.")


# Define the root directory and regex pattern
root_dir = "./"



cleaned_runtime_data = {}

acc_data = {}
for impl in ['dask', 'index']:
# Iterate through subdirectories of root_dir
   
    for entry in os.listdir(root_dir):

        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path) and impl in full_path:
            # print(f"Matched directory: {full_path}")


            if os.path.isfile(f"{full_path}/stats.csv"):
                stats = pd.read_csv(f"{full_path}/stats.csv")
                key = f"{impl}-{extract_nodes(full_path)}"
                # print(stats)
                if impl =="dask":
                    
                    row = [float(stats['total'])]
                else:
                    row = [float(stats['total'])]
                    
                    stats = pd.read_csv(f"{full_path}/per_epoch_stats.csv")
                    t_mae = list(stats['t_mae'])
                    v_mae = list(stats['v_mae'])
                    
                    acc_data[key] = [t_mae, v_mae]
                cleaned_runtime_data[key] = row

workers = ["4", "8", "16", "32", "64", "128"]

# Creating subplots (1 row, 3 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
plt.rcParams.update({'font.size': 14})

ax1.set_title("DDP", fontweight='bold')


dask_comp = [
    cleaned_runtime_data['dask-1'][0],  
    cleaned_runtime_data['dask-2'][0],
    cleaned_runtime_data['dask-4'][0],
    cleaned_runtime_data['dask-8'][0],
    cleaned_runtime_data['dask-16'][0],
    cleaned_runtime_data['dask-32'][0],
    ]
dask_comp = np.array(dask_comp) / 60

bars1a = ax1.bar(workers, dask_comp, color="tab:blue")
ax1.set_xlabel('GPUs')
ax1.set_ylabel('Minutes', labelpad=10)


ax2.set_title("Distributed-Index-Batching", fontweight='bold')

index_comp = [
    cleaned_runtime_data['index-1'][0],  
    cleaned_runtime_data['index-2'][0],
    cleaned_runtime_data['index-4'][0],
    cleaned_runtime_data['index-8'][0],
    cleaned_runtime_data['index-16'][0],
    cleaned_runtime_data['index-32'][0],
    ]
index_comp = np.array(index_comp) / 60

bars2 = ax2.bar(workers, index_comp, color="tab:blue")

gpu_baseline = gpu_baseline / 60
ax2.plot(workers, [ gpu_baseline/ 4, gpu_baseline/ 8, gpu_baseline/ 16, 
                   gpu_baseline/ 32, gpu_baseline/ 64, gpu_baseline/ 128], 
         linestyle="dashed", label="Linear Scaling", marker="o", color="black")

ax2.set_xlabel('GPUs')
ax2.set_ylabel('Minutes', labelpad=10)

# # Set custom y-ticks for all plots
desired_ticks = [i for i in range(0,180, 20)]
for ax in [ax1]:
    ax.set_yticks(desired_ticks)
    ax.set_yticklabels([str(tick) for tick in desired_ticks], fontsize=14)

ax2.tick_params(labelleft=True)
for ax in [ax2]:
    ax.set_yticks(desired_ticks)
    ax.set_yticklabels([str(tick) for tick in desired_ticks], fontsize=14)

plt.tight_layout()
plt.savefig("figure7.pdf",format='pdf', bbox_inches='tight')
plt.show()
