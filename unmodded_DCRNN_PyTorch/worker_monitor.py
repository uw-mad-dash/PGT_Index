import pickle
import scipy.sparse as sp
import torch
import numpy as np
import pandas as pd
import csv
import os
import time
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

def collect_metrics():
    
    try:
        # Initialize NVML for GPU metrics
        nvmlInit()

        # Open the CSV file in append mode
        
        data = []
        max_gpu_mem = -1
        max_system_mem = -1
        max_gpu_mem2 = -1
        max_system_mem2 = -1
        while True:
            # Collect system memory usage
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            mem = psutil.virtual_memory()

            total_rss = sum(proc.memory_info().rss for proc in psutil.process_iter(attrs=['memory_info']))
            system_memory_used = total_rss / (1024**2)  # Convert to MB
            system_memory_total = mem.total / (1024**2)  # Convert to MB

            # Collect GPU memory usage
            gpu_metrics = []
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used = info.used / (1024**2)  # Convert to MB
            gpu_memory_total = info.total / (1024**2)  # Convert to MB

            max_gpu_mem = max(gpu_memory_used, max_gpu_mem)
            max_system_mem = max(system_memory_used, max_system_mem)
            
            handle2 = nvmlDeviceGetHandleByIndex(1)
            info2 = nvmlDeviceGetMemoryInfo(handle2)
            gpu_memory_used2 = info2.used / (1024**2)  # Convert to MB
    

            # max_gpu_mem2 = max(gpu_memory_used2, max_gpu_mem2)

            data.append([timestamp,system_memory_used,system_memory_total, gpu_memory_used, gpu_memory_used2, gpu_memory_total])
            
            

           
            with open("monitor_system_stats.csv", mode="w", newline="") as f:
                writer = csv.writer(f)

                # Write headers to the CSV file
                headers = [
                    "Timestamp",
                    "System_Memory_Used",
                    "System_Memory_Total",
                    "GPU_0_Memory_Used",
                    "GPU_1_Memory_Used",
                    "GPU_Memory_Total"
                ]
                writer.writerow(headers)
                writer.writerows(data)
            time.sleep(1)

    except Exception as e:
        print("Error in collecting metrics:", str(e))

    finally:
        # Shutdown NVML
        nvmlShutdown()



collect_metrics()