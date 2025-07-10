import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

datasets = ['Chickenpox', 'Windmill', 'PeMS-Bay']

# Epochs
val_epochs = [np.arange(1, 101), np.arange(1, 101), np.arange(1, 101)]

# Load validation data
val_baseline = [
    pd.read_csv("chickenPoxBase1/per_epoch_stats.csv")['v_mae'],
    pd.read_csv("WindmillBase1/per_epoch_stats.csv")['v_mae'],
    pd.read_csv("PemsBayBase1/per_epoch_stats.csv")['v_mae'],
]

val_index = [
    pd.read_csv("chickenPoxIndex1/per_epoch_stats.csv")['v_mae'],
    pd.read_csv("WindmillIndex1/per_epoch_stats.csv")['v_mae'],
    pd.read_csv("PemsBayIndex1/per_epoch_stats.csv")['v_mae'],
]

# Set up 1 row x 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

for i in range(3):
    axs[i].plot(val_epochs[i], val_baseline[i], label='Baseline', marker='o', markevery=5)
    axs[i].plot(val_epochs[i], val_index[i], label='Index', marker='x', markevery=5)
    axs[i].set_title(f'{datasets[i]} - Validation MAE', fontweight="heavy", fontsize=16)
    axs[i].grid(True)
    axs[i].legend(fontsize=12)

    if i == 0:
        axs[i].set_xlabel('Epoch', fontsize=14)
        axs[i].set_ylabel('MAE', fontsize=14, labelpad=10)

plt.tight_layout()
plt.savefig("figure5.pdf", format='pdf', bbox_inches='tight')
plt.show()