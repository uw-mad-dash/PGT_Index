from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import time
from lib.utils import load_graph_data
from model.pytorch.dcrnn_sup_with_prof import DCRNNSupervisor
import pynvml
import csv
import time
import psutil
import threading

global stop

def monitor_local_thread():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # Open the CSV file in append mode ('a'), ensuring data is added in case of shutdown or restart
    with open(f"worker_{0}_stats.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file is empty, write the header
        file.seek(0, 2)  # Go to the end of the file to check if it's empty
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'System_MEM','USED_SYSTEM_MEM','SYSTEM_CPU%', 'cpu_mem', 'cpu_util', 'r_count', 'r_bytes', 'w_count', 'w_byes', 'GPU_MEM', 'Free_GPU_MEM', 'GPU_USED_MEM', 'GPU_util', 'GPU_mem_util', 'GPU_temp', 'GPU_mW'])

        
        while stop:
            system_memory = psutil.virtual_memory()
            system_cpu_percent = psutil.cpu_percent(interval=1)
            cpu_mem = system_memory.total
            cpu_used_mem = system_memory.used
            current_process = psutil.Process()
            memory_info = current_process.memory_info()

            # RSS (Resident Set Size) is the actual memory the process is using in bytes
            rss_memory = memory_info.rss
            

            cpu_percent = current_process.cpu_percent(interval=1)  # interval=1 to get the average over 1 second
            
            
            io_counters = current_process.io_counters()
            read_count = io_counters.read_count
            read_bytes = io_counters.read_bytes
            write_count = io_counters.write_count
            write_bytes = io_counters.write_bytes


            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total    # Raw value in bytes
            free_memory = memory_info.free      # Raw value in bytes
            used_memory = memory_info.used      # Raw value in bytes
            
            # Get utilization rates (GPU and memory utilization)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu        # GPU utilization in percentage
            memory_utilization = utilization.memory  # Memory utilization in percentage
            
            # Get temperature (in Celsius)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Get power usage (in milliwatts, raw)
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)  # Raw value in milliwatts
            
            # Get the current timestamp
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Write the stats to the CSV file
            writer.writerow([timestamp, cpu_mem, cpu_used_mem, system_cpu_percent,rss_memory, cpu_percent, read_count, read_bytes, write_count, write_bytes, total_memory, free_memory, used_memory, gpu_utilization, 
                            memory_utilization, temperature, power_usage])
                            
           
            # Flush the data to the file (to ensure data is saved after each iteration)
            file.flush()
            
            # Sleep for 1 second before getting the next reading
            time.sleep(1)

        

    # Shutdown NVML when done
    print("Exiting",flush=True)
    pynvml.nvmlShutdown()

def main(args):
    with open(args.config_filename) as f:
        global stop
        stop = True
        thread = threading.Thread(target=monitor_local_thread)
        thread.start()

        complete_start = time.time()
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train(complete_start)
        stop = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
