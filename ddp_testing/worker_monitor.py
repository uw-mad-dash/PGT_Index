import csv
import time
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
import os

def collect_metrics():
    try:
        # Initialize NVML for GPU metrics
        nvmlInit()

        # Open the CSV file in append mode
        data = []
        with open("system_stats.csv", mode="w", newline="") as f:
                writer = csv.writer(f)

                # Write headers to the CSV file
                headers = [
                    "Timestamp",
                    "System_Memory_Used",
                    "System_Memory_Total",
                    "GPU_Memory_Used",
                    "GPU_Memory_Total"
                ]
                writer.writerow(headers)

        while True:
            # Collect system memory usage
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            process = psutil.Process()
            rss = sum(proc.memory_info().rss for proc in psutil.process_iter(attrs=['memory_info']))

            mem = psutil.virtual_memory()
            system_memory_used = rss / (1024**2)  # Convert to MB
            system_memory_total = mem.total / (1024**2)  # Convert to MB

            # Collect GPU memory usage
            
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used = info.used / (1024**2)  # Convert to MB
            gpu_memory_total = info.total / (1024**2)  # Convert to MB

            data.append([timestamp,system_memory_used,system_memory_total, gpu_memory_used, gpu_memory_total])
            
            with open("system_stats.csv", mode="a", newline="") as f:
                writer = csv.writer(f)

                
                writer.writerow([timestamp,system_memory_used,system_memory_total, gpu_memory_used, gpu_memory_total])

            if os.path.isfile("flag.txt"):
                os.remove("flag.txt")
                break
  
            
            time.sleep(1)

    except Exception as e:
        print("Error in collecting metrics:", str(e))

    finally:
        # Shutdown NVML
        nvmlShutdown()

collect_metrics()