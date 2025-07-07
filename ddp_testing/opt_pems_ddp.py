import time 
import csv
import argparse
import uuid
import os
import random
import math

from torch_geometric_temporal.nn.recurrent import BatchedDCRNN as DCRNN
from torch_geometric_temporal.dataset import IndexDataset,DaskDataset
from utils import *


import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler, DataLoader, Dataset
from dask.array.optimization import optimize


from dask.distributed import LocalCluster
from dask.distributed import Client
from dask_pytorch_ddp import dispatch, results
from dask.distributed import wait as Wait


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument(
        "-m", "--mode", type=str, default="dask", help="Which version to run"
    )
    parser.add_argument(
        "-g", "--gpu", type=str, default="False", help="Should data be preprocessed and migrated directly to the GPU"
    )
    parser.add_argument(
        "-d", "--debug", type=str, default="False", help="Print values for debugging"
    )
    parser.add_argument(
     "--dist", type=str, default="False", help="Is computation distributed across multiple nodes"
    )

    parser.add_argument(
     "-np","--npar", type=int, default=1, help="The number of GPUs/workers per node"
    )
    parser.add_argument(
     "--dataset", type=str, default="pems-bay", help="Which dataset is in use"
    )
    
    return parser.parse_args()


def train(args=None, epochs=None, batch_size=None, allGPU=False, debug=False,
         x_train=None, y_train=None, x_val=None, y_val=None, mean=None, std=None,
         data=None,
         global_start=None,pre_t1=None, pre_t2=None):

    # train_dataloader, val_dataloader, mean, std, epochs, seq_length, num_nodes, num_features,
    worker_rank = int(dist.get_rank())
    gpu = worker_rank % 4   
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()

    
    
    if args.dataset == "pems-bay":
        filepath = "data/pems-bay.h5"
        adj_filepath = "data/adj_mx_bay.pkl"
        key = "speed"
    
    elif args.dataset.lower() == "pems-all-la":
        filepath = "data/pems-all-la.h5"
        adj_filepath = "data/adj_mx_all_la.pkl"
        key = "df"
    
    elif args.dataset.lower() == "pems":
        filepath = "data/pems.h5"
        adj_filepath = "data/adj_mx_pems.pkl"
        key = "df"
    
    else:
        print("Invalid dataset option. The currently supported datasets are: 'pems-bay', 'pems-all-la', and 'pems'")
        return 
    
    if args.mode == "dask":
        mean = torch.tensor(mean.compute()).to(device)
        std = torch.tensor(std.compute()).to(device)
        x_train_idx = np.arange(0, x_train.shape[0])
        x_val_idx = np.arange(x_train.shape[0], x_train.shape[0] + x_val.shape[0])
        
    
    if args.mode == "dask-index":
        data_size = data.shape[0] - (2 * 12 - 1)
       
        x_i = np.arange(data_size)
        num_samples = x_i.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.7)
        num_val = num_samples - num_test - num_train

        x_train_idx = x_i[:num_train]
        x_val_idx = x_i[num_train: num_train + num_val]
       

        mean = torch.tensor(mean.compute()).to(device)
        std = torch.tensor(std.compute()).to(device)
        
    
    
    
    
    pre_t2 = time.time()
    if worker_rank == 0:
        print("Preprocessing: ", pre_t2 - pre_t1, flush=True)
    

    train_t1 = time.time()
    

    # Load graph structure
    _, _, adj_mx = load_graph_data(adj_filepath)
    edge_index, edge_weight = adjacency_to_edge_index(adj_mx)

    # Move to GPU
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    # Initialize model
    model = DCRNN(2, 2, K=3).to(device)
    model = DDP(model, gradient_as_bucket_view=True, device_ids=[device], output_device=[device])

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    
    train_dataset = CopyDataset(x_train_idx)
    val_dataset = CopyDataset(x_val_idx)
    
    train_sampler = EvenChunkDistributedSampler(train_dataset, num_replicas=world_size, rank=worker_rank, shuffle=False) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = EvenChunkDistributedSampler(val_dataset, num_replicas=world_size, rank=worker_rank, shuffle=False)                  
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    

    # Training loop
    stats = []
    min_t = 9999
    min_v = 9999
    for epoch in range(epochs):
        epoch_start = time.time()
        
        total = len(train_dataloader)
        
        model.train()
        train_loss = 0.0
        i = 1

        train_b = [np.array(batch) for batch in train_dataloader]
        random.shuffle(train_b)
        

        for batch in train_b:

            if args.mode == "dask":
                t1 = time.time()
                X_batch, y_batch = torch.tensor(x_train[batch].compute(), dtype=torch.float32), torch.tensor(y_train[batch].compute(),dtype=torch.float32)

                t2 = time.time()
                f1 = time.time()
                batch_items = len(batch)
            
            if args.mode == "dask-index":

                extend = batch[-1] + 1
                batch = np.concatenate((batch, np.arange(extend, extend + 23)))
                
                t1 =time.time()
                batch_data = torch.tensor(data[batch].compute())
                t2 = time.time()
                f1 = time.time()

                batch_items = len(batch) -23
                
                X_batch, y_batch = torch.empty(batch_items, 12, data.shape[1], data.shape[2]), torch.empty(batch_items, 12, data.shape[1], data.shape[2])
                for j in range(batch_items):
                    
                    X_batch[j] = batch_data[j: j+12]
                    y_batch[j] = batch_data[j + 12: j+24]
                
            
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            # Forward pass
            outputs = model(X_batch, edge_index, edge_weight)  # Shape: (batch_size, seq_length, num_nodes, out_channels)

            # Calculate loss (use only the first output channel, assuming it's the target)
            loss = masked_mae_loss((outputs * std) + mean, (y_batch * std) + mean)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            f2 = time.time()

          
            if debug and worker_rank == 0:
                print(f"Train Batch: {i}/{total}", end="\r", flush="True")
                i += 1
            

        train_loss /= len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        total = len(val_dataloader)
        
        if debug and worker_rank == 0:
            print("                                   ",end="\r")

        i = 1
        with torch.no_grad():
            for batch in val_dataloader:
                batch = np.array(batch)
                
                if args.mode == "dask":
                    batch = batch - x_train.shape[0]

                    t1 = time.time()
                    X_batch, y_batch = torch.tensor(x_val[batch].compute()), torch.tensor(y_val[batch].compute())
                    t2 = time.time()
                    f1 = time.time()
                    batch_items = len(batch)
                if args.mode == "dask-index":

                    extend = batch[-1] + 1
                    batch = np.concatenate((batch, np.arange(extend, extend + 23)))
                    
                    t1 =time.time()
                    batch_data = torch.tensor(data[batch].compute())
                    t2 = time.time()
                    f1 = time.time()

                    batch_items = len(batch) -23
                    
                    X_batch, y_batch = torch.empty(batch_items, 12, data.shape[1], data.shape[2]), torch.empty(batch_items, 12, data.shape[1], data.shape[2])
                    for j in range(batch_items):
                        
                        X_batch[j] = batch_data[j: j+12]
                        y_batch[j] = batch_data[j + 12: j+24]

                
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

                # Forward pass
                outputs = model(X_batch, edge_index, edge_weight)  # Shape: (batch_size, seq_length, num_nodes, out_channels)

                # Calculate loss (use only the first output channel, assuming it's the target)
                loss = masked_mae_loss((outputs * std) + mean, (y_batch * std) + mean)

                

                val_loss += loss.item()
                f2 = time.time()
                if debug and worker_rank == 0:
                    print(f"Val Batch: {i}/{total}", end="\r", flush=True)
                    i += 1
                

        # # average valdiation across all ranks
        val_tensor = torch.tensor([val_loss, len(val_dataloader)])
        dist.reduce(val_tensor,dst=0, op=dist.ReduceOp.SUM)
        epoch_end = time.time()
        

        if worker_rank == 0:
            val_loss = val_tensor[0]/ val_tensor[1]
            # Print epoch metrics
            print(f"Epoch {epoch + 1}/{epochs}, Runtime: {epoch_end - epoch_start}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)
            stats.append([epoch+1, epoch_end - epoch_start, train_loss, float(val_loss)])

            min_t = min(min_t, train_loss)
            min_v = min(min_v, val_loss)
        
    train_t2 = time.time()
    if worker_rank == 0:
        with open("per_epoch_stats.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch","runtime","t_mae", "v_mae"])
            writer.writerows(stats)
    
    
        with open("stats.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["preprocessing","training","total", "t_mae", "v_mae"])
            writer.writerow([pre_t2 - pre_t1, train_t2 - train_t1, train_t2 - global_start, min_t, float(min_v)])


def main():
  


    args = parse_arguments()
    allGPU = args.gpu.lower() == "true"
    dist = args.dist.lower() == "true"
    npar = args.npar
    debug = args.debug.lower() == "true"
    print(args.mode,allGPU, flush=True)
    batch_size = 64
    epochs = 30
    
    if not os.path.isdir("data"):
        raise FileNotFoundError("Error: The 'data/' subdirectory is missing. "
                                "Scripts assume data and adjacency matrix files are in 'data/'.")
    global_start = time.time()
    if dist:
        client = Client(scheduler_file = f"cluster.info")
    else:
        cluster = LocalCluster(n_workers=npar)
        client = Client(cluster)

    if args.mode == "dask":
        
        if args.dataset == "pems-bay":
            filepath = "data/pems-bay.h5"
            adj_filepath = "data/adj_mx_bay.pkl"
            key = "speed"
        
        elif args.dataset.lower() == "pems-all-la":
            filepath = "data/pems-all-la.h5"
            adj_filepath = "data/adj_mx_all_la.pkl"
            key = "df"
        
        elif args.dataset.lower() == "pems":
            filepath = "data/pems.h5"
            adj_filepath = "data/adj_mx_pems.pkl"
            key = "df"
        
        else:
            print("Invalid dataset option. The currently supported datasets are: 'pems-bay', 'pems-all-la', and 'pems'")
            return 
        
        pre_t1 = time.time()
        x_train, y_train, x_val, y_val, x_test, y_test, mean, std = DaskDataset.preprocess(filepath, key)
        
        mean, std, x_train, y_train, x_val, y_val, x_test, y_test = client.persist([mean, std, x_train, y_train, x_val, y_val,  x_test, y_test])  
        Wait([mean, std, x_train, y_train, x_val, y_val, x_test, y_test])
        
        if dist:
            for f in ['utils.py', "opt_dask.py"]:
                print("Uploading ", f, flush=True)
                client.upload_file(f)

        futures = dispatch.run(client, train,
                            args=args, epochs=epochs, debug=debug, batch_size=batch_size,
                            global_start=global_start, pre_t1=pre_t1,
                            x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, mean=mean, std=std,
                            backend="gloo")

    
    elif args.mode == "dask-index":
        if args.dataset == "pems-bay":
            filepath = "data/pems-bay.h5"
            adj_filepath = "data/adj_mx_bay.pkl"
            key = "speed"
        
        elif args.dataset.lower() == "pems-all-la":
            filepath = "data/pems-all-la.h5"
            adj_filepath = "data/adj_mx_all_la.pkl"
            key = "df"
        
        elif args.dataset.lower() == "pems":
            filepath = "data/pems.h5"
            adj_filepath = "data/adj_mx_pems.pkl"
            key = "df"
        
        else:
            print("Invalid dataset option. The currently supported datasets are: 'pems-bay', 'pems-all-la', and 'pems'")
            return 
        
        pre_t1 = time.time() 

        data, mean, std = IndexDataset.dask_preprocess(filepath,key,12, add_time_in_day=True, target_chunk_size=(12,11160,2))
        data, mean, std, = client.persist([data, mean, std])
        Wait([data, mean, std])

        if dist:
            for f in ['utils.py', "opt_dask.py"]:
                print("Uploading ", f, flush=True)
                client.upload_file(f)
        futures = dispatch.run(client, train,
                            args=args, debug=debug, epochs=epochs, batch_size=batch_size,
                            global_start=global_start, pre_t1=pre_t1,
                            data=data, mean=mean, std=std,
                            backend="gloo")
    else:
        print("Invalid mode; enter 'dask', or 'dask-index'")
        open("flag.txt", "w").close()
        time.sleep(1)
        return


    key = uuid.uuid4().hex
    rh = results.DaskResultsHandler(key)
    rh.process_results(".", futures, raise_errors=False)
    client.shutdown()

    open("flag.txt", "w").close()
    time.sleep(1) 
    
if __name__ == "__main__":
    main()