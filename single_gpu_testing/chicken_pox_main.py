import time 
import csv

from torch_geometric_temporal.nn.recurrent import BatchedDCRNN as DCRNN
from torch_geometric_temporal.dataset import IndexDataset
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import argparse

from utils import *
import os

import threading
from torch_geometric_temporal.dataset import seq_seq_ChickenpoxDatasetLoader

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



def train(train_dataloader, val_dataloader, edge_index,edge_weight, epochs, seq_length, num_nodes, num_features, 
            allGPU=False, debug=False):
    
    
    

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

            # Calculate loss
            loss = masked_mae_loss(outputs,y_batch )
           
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

                # Calculate loss
                loss = masked_mae_loss(outputs,y_batch)
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
    debug = args.debug.lower() == "true"
    print(args.mode,allGPU, flush=True)
    batch_size = 4
    epochs = 100



    if not os.path.isdir("data"):
        raise FileNotFoundError("Error: The 'data/' subdirectory is missing. "
                                "Scripts assume data and adjacency matrix files are in 'data/'.")



    thread = threading.Thread(target=collect_metrics)
    thread.start()

    if args.mode == "base":
        # preprocess data
        pre_t1 = time.time()
        # Load the dataset
        loader = seq_seq_ChickenpoxDatasetLoader()
        loader.get_dataset()
               
        x_tensor = torch.tensor(loader.features, dtype=torch.float)
        y_tensor = torch.tensor(loader.targets, dtype=torch.float)

        
        
        edges = torch.tensor(loader._edges,dtype=torch.int64)
        edge_weights = torch.tensor(loader._edge_weights,dtype=torch.float)

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
        
        pre_t2 = time.time()
        print("Preprocessing: ", pre_t2 - pre_t1, flush=True)
        
        del x_tensor, y_tensor, x_test, y_test
        train_t1 = time.time()

        if allGPU:
            x_train = x_train.to("cuda:0")
            y_train = y_train.to("cuda:0")
            x_val = x_val.to("cuda:0")
            y_val = y_val.to("cuda:0")
            
        
        # Create Datasets and DataLoaders
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        t_min, v_min = train(train_dataloader, val_dataloader, edges, edge_weights, epochs, 4,20,1, allGPU=allGPU, debug=debug)
        train_t2 = time.time()
        print("Training: ", train_t2 - train_t1, flush=True)

    elif args.mode == "index":
        pre_t1 = time.time()
        
        if allGPU == True:
            data, x_train, x_val, x_test, edges, edge_weights = IndexDataset.preprocess("",4,"chickenpox", gpu=0)
        else:
            data, x_train, x_val, x_test, edges, edge_weights = IndexDataset.preprocess("",4,"chickenpox")
        

        edges = torch.tensor(edges,dtype=torch.int64)
        edge_weights = torch.tensor(edge_weights,dtype=torch.float)
        
        pre_t2 = time.time()
        print("Preprocessing: ", pre_t2 - pre_t1, flush=True)    
        train_t1 = time.time()

        train_dataset = IndexDataset(x_train,data,4,gpu=allGPU)
        val_dataset = IndexDataset(x_val,data,4,gpu=allGPU)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        
        t_min, v_min = train(train_dataloader, val_dataloader, edges, edge_weights, epochs, 4,20,1, allGPU=allGPU, debug=debug)

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