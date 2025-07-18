import time 
import csv

import torch
from torch_geometric_temporal.nn.recurrent import BatchedDCRNN as DCRNN
from torch_geometric_temporal.dataset import IndexDataset
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import argparse

from utils import *
import threading
import os
import requests
from tqdm import tqdm

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument(
        "-m", "--mode", type=str, default="base", help="Which version to run"
    )
    parser.add_argument(
        "-g", "--gpu", type=str, default="False", help="Should data be preprocessed and migrated directly to the GPU"
    )
    parser.add_argument(
        "-d", "--debug", type=str, default="False", help="Print values for debugging"
    )
    
    return parser.parse_args()


def downloadCheck():
    if not os.path.isdir("data"):
        raise FileNotFoundError("Error: The 'data/' subdirectory is missing. "
                                "Scripts assume data and adjacency matrix files will be placed in 'data/'.")
    else:
        PeMS_file_links = {
        "adj_mx_pems.pkl" : "https://anl.app.box.com/shared/static/4143x1repqa1u26aiz7o2rvw3vpcu0wp",
        "pems.h5": "https://anl.app.box.com/shared/static/7hfhtie02iufy75ac1d8g8530majwci0"
        }          
        
        for key in PeMS_file_links.keys():
            if not os.path.isfile(os.path.join("./data/",key)):
                print("Downloading ", key, flush=True)
                response = requests.get(PeMS_file_links[key], stream=True)
                file_size = int(response.headers.get('content-length', 0))

                with open(os.path.join("./data/", key), "wb") as file, tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as progress_bar:
                    for chunk in response.iter_content(chunk_size=33554432):
                        file.write(chunk)
                        progress_bar.update(len(chunk))

def train(train_dataloader, val_dataloader, mean, std, epochs, seq_length, num_nodes, num_features, allGPU=False, debug=False):
    
    
    # Load graph structure
    sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data("data/adj_mx_pems.pkl")
    edge_index, edge_weight = adjacency_to_edge_index(adj_mx)

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    # Initialize model
    model = DCRNN(num_features, num_features, K=3).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    stats = []
    min_t = 9999
    min_v = 9999
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        i = 1
        total = len(train_dataloader)
        t1 = time.time()
        for batch in train_dataloader:
            X_batch, y_batch = batch

            if allGPU == False:
                # print("casting")
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

            # Forward pass
            outputs = model(X_batch, edge_index, edge_weight)  # Shape: (batch_size, seq_length, num_nodes, out_channels)

            """
            To match the open-source preprocessing implementation 
            (https://github.com/liyaguang/DCRNN/blob/master/lib/utils.py#L178),
            only the first feature (speed) is normalized. We reverse this standardization 
            before computing MAE.
            """
            outputs[...,0] = (outputs[...,0] * std) + mean
            y_batch[...,0] = (y_batch[...,0] * std) + mean
            
            # Calculate loss
            loss = masked_mae_loss(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if debug:
                print(f"Train Batch: {i}/{total}", end="\r")
                i+=1
            

        train_loss /= len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        i = 0
        if debug:
            print("                      ", end="\r")
        total = len(val_dataloader)
        with torch.no_grad():
            for batch in val_dataloader:
                X_batch, y_batch = batch

                if allGPU == False:
                    X_batch = X_batch.to(device).float()
                    y_batch = y_batch.to(device).float()

                # Forward pass
                outputs = model(X_batch, edge_index, edge_weight)

                outputs[...,0] = (outputs[...,0] * std) + mean
                y_batch[...,0] = (y_batch[...,0] * std) + mean
            
                # Calculate loss
                loss = masked_mae_loss(outputs, y_batch)
                val_loss += loss.item()
                if debug:
                    print(f"Val Batch: {i}/{total}", end="\r")
                    i += 1
                

        val_loss /= len(val_dataloader)
        t2 = time.time()
        # Print epoch metrics
        print(f"Epoch {epoch + 1}/{epochs}, Runtime: {t2 - t1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)
        stats.append([epoch+1, t2 - t1, train_loss, val_loss])

        min_t = min(min_t, train_loss)
        min_v = min(min_v, val_loss)

    with open("per_epoch_stats.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch","runtime","t_mae", "v_mae"])
        writer.writerows(stats)
    
    return min_t, min_v

def main():
  


    args = parse_arguments()
    allGPU = args.gpu.lower() == "true"
    print(args.mode,allGPU, flush=True)
    debug = args.debug.lower() == "true"
    batch_size = 64
    epochs = 30

    downloadCheck()
    
    thread = threading.Thread(target=collect_metrics, kwargs={'failsafe': True})
    thread.start()

    if args.mode == "base":
        # preprocess data
        pre_t1 = time.time()
        x_tensor,y_tensor = benchmark_preprocess("data/pems.h5","pems")

        num_samples = x_tensor.shape[0]
        num_train = round(num_samples * 0.7)
        num_test = round(num_samples * 0.2)
        num_val = num_samples - num_train - num_test

        x_train = x_tensor[:num_train]
        y_train = y_tensor[:num_train]
        x_val = x_tensor[num_train: num_train + num_val]
        y_val = y_tensor[num_train: num_train + num_val]
        x_test = x_tensor[-num_test:]
        y_test = y_tensor[-num_test:]

        mean = x_train[...,0].mean()
        std = x_train[...,0].std()

        x_train[...,0] = (x_train[...,0] - mean) / std 
        y_train[...,0] = (y_train[...,0] - mean) / std

        x_val[...,0] = (x_val[...,0] - mean) / std 
        y_val[...,0] = (y_val[...,0] - mean) / std

        x_test[...,0] = (x_test[...,0] - mean) / std 
        y_test[...,0] = (y_test[...,0] - mean) / std
        
        del x_tensor, y_tensor, x_test, y_test


        pre_t2 = time.time()
        print("Preprocessing: ", pre_t2 - pre_t1, flush=True)
        
        train_t1 = time.time()

        if allGPU:
            x_train = x_train.to("cuda:0")
            y_train = y_train.to("cuda:0")
            x_val = x_val.to("cuda:0")
            y_val = y_val.to("cuda:0")
        # Create Datasets and DataLoaders
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        t_min, v_min = train(train_dataloader, val_dataloader, mean, std, epochs, 12,11160,2, allGPU=allGPU, debug=debug)
        train_t2 = time.time()
        print("Training: ", train_t2 - train_t1, flush=True)

    elif args.mode == "index":
        pre_t1 = time.time()
        
        if allGPU == True:
            data, mean, std, x_train, x_val, x_test = IndexDataset.preprocess("data/pems.h5", 12,"pems", h5_key="df", add_time_in_day=True,gpu=0)
        else:
            data, mean, std, x_train, x_val, x_test = IndexDataset.preprocess("data/pems.h5",12,"pems", h5_key="df",add_time_in_day=True)

        pre_t2 = time.time()
        print("Preprocessing: ", pre_t2 - pre_t1, flush=True)

        train_t1 = time.time()
        
        train_dataset = IndexDataset(x_train,data,12,gpu=allGPU)
        val_dataset = IndexDataset(x_val,data,12,gpu=allGPU)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        
        t_min, v_min = train(train_dataloader, val_dataloader, mean, std, epochs, 12,11160,2, allGPU=allGPU, debug=debug)
        train_t2 = time.time()
        print("Training: ", train_t2 - train_t1, flush=True)
    else:
        print("Invalid mode; enter either 'index' or 'base'")
        open("flag.txt", "w").close()
        time.sleep(1)
        return 

    with open("stats.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["preprocessing","training","total", "t_mae", "v_mae"])
        writer.writerow([pre_t2 - pre_t1, train_t2 - train_t1, (pre_t2 - pre_t1)+ (train_t2 - train_t1), t_min, v_min])

if __name__ == "__main__":
    main()