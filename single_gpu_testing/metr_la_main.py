import numpy as np
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from torch_geometric_temporal.dataset import METRLADatasetLoader
import argparse
from utils import *
import threading

def parse_arguments():
    parser = argparse.ArgumentParser(description="Demo of index batching with PemsBay dataset")

    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="The desired number of training epochs"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="index", help="Which version to run"
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32, help="The desired batch size"
    )
    parser.add_argument(
        "-g", "--gpu", type=str, default="False", help="Should data be preprocessed and migrated directly to the GPU"
    )
    parser.add_argument(
        "-d", "--debug", type=str, default="False", help="Print values for debugging"
    )
    return parser.parse_args()

# Making the model 
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, periods=periods,batch_size=batch_size) # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h) 
        h = self.linear(h)
        return h



def train(train_dataloader, val_dataloader, batch_size, epochs, edges, DEVICE, allGPU=False, debug=False, mode="index"):
    
    # Create model and optimizers
    model = TemporalGNN(node_features=2, periods=12, batch_size=batch_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    stats = []
    t_mse = []
    v_mse = []

    
    edges = edges.to(DEVICE)
    for epoch in range(epochs):
        step = 0
        loss_list = []
        t1 = time.time()
        i = 1
        total = len(train_dataloader)
        for batch in train_dataloader:
            X_batch, y_batch = batch
            
            # Need to permute based on expected input shape for ATGCN
            if mode == "index":
                if allGPU:
                    X_batch = X_batch.permute(0, 2, 3, 1)
                    y_batch = y_batch[...,0].permute(0, 2, 1)
                else:
                    X_batch = X_batch.permute(0, 2, 3, 1).to(DEVICE)
                    y_batch = y_batch[...,0].permute(0, 2, 1).to(DEVICE)
            else:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)


            y_hat = model(X_batch, edges)         # Get model predictions
            loss = loss_fn(y_hat, y_batch) # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step= step+ 1
            loss_list.append(loss.item())

            if debug:
                print(f"Train Batch: {i}/{total}", end="\r")
                i+=1


        model.eval()
        step = 0
        # Store for analysis
        total_loss = []
        i = 1
        total = len(val_dataloader)
        if debug:
            print("                      ", end="\r")
        with torch.no_grad():
            for batch in val_dataloader:
                X_batch, y_batch = batch

                
                # Need to permute based on expected input shape for ATGCN
                if mode == "index":
                    if allGPU:
                        X_batch = X_batch.permute(0, 2, 3, 1)
                        y_batch = y_batch[...,0].permute(0, 2, 1)
                    else:
                        X_batch = X_batch.permute(0, 2, 3, 1).to(DEVICE)
                        y_batch = y_batch[...,0].permute(0, 2, 1).to(DEVICE)
                else:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                # Get model predictions
                y_hat = model(X_batch, edges)
                # Mean squared error
                loss = loss_fn(y_hat, y_batch)
                total_loss.append(loss.item())
                
                if debug:
                    print(f"Val Batch: {i}/{total}", end="\r")
                    i += 1
                
            
        t2 = time.time()
       

        print("Epoch {} time: {:.4f} train RMSE: {:.4f} Test MSE: {:.4f}".format(epoch,t2 - t1, sum(loss_list)/len(loss_list), sum(total_loss)/len(total_loss)))
        stats.append([epoch, t2-t1, sum(loss_list)/len(loss_list), sum(total_loss)/len(total_loss)])
        t_mse.append(sum(loss_list)/len(loss_list))
        v_mse.append(sum(total_loss)/len(total_loss))
    
    with open("per_epoch_stats.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch","runtime","t_mse", "v_mse"])
        writer.writerows(stats)
        
    return min(t_mse), min(v_mse)
        

  





def main():
    args = parse_arguments()
    allGPU = args.gpu.lower() in ["true", "y", "t", "yes"]
    debug = args.debug.lower() in ["true", "y", "t", "yes"]
    batch_size = args.batch_size
    epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffle= True

    start = time.time()
    if args.mode.lower() == "index":
        thread = threading.Thread(target=collect_metrics)
        thread.start()

        p1 = time.time() 
        indexLoader = METRLADatasetLoader(index=True)

        # To mimic the 80/20 train/test split in the original a3T-GCN script, we omit a validation set
        # The index loader for metraLA is hard-coded to drop_last to mirror the original
        if allGPU:
            train_dataloader, _, test_dataloader, edges, edge_weights, mean, std = indexLoader.get_index_dataset(batch_size=batch_size, shuffle=shuffle, allGPU=0, ratio=(0.8,0.0,0.2)) 
        else:
            train_dataloader, _, test_dataloader, edges, edge_weights, mean, std = indexLoader.get_index_dataset(batch_size=batch_size, shuffle=shuffle, ratio=(0.8,0.0,0.2)) 
        p2 = time.time()
        train_t1 = time.time() 
        t_mse, v_mse = train(train_dataloader, test_dataloader, batch_size, epochs, edges, device, debug=debug)
        end = time.time()
    elif args.mode.lower() == "base":
        thread = threading.Thread(target=collect_metrics)
        thread.start()

        # baseline taken from https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/recurrent/a3tgcn2_example.py
        p1 = time.time()
        loader = METRLADatasetLoader(raw_data_dir="./data/")
        dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
        
        # Train test split 
        from torch_geometric_temporal.signal import temporal_signal_split
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)


        # Creating Dataloaders
        train_input = np.array(train_dataset.features) # (27399, 207, 2, 12)
        train_target = np.array(train_dataset.targets) # (27399, 207, 12)
        train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor)  # (B, N, F, T)
        train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor)  # (B, N, T)
        train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle,drop_last=True)


        test_input = np.array(test_dataset.features) # (, 207, 2, 12)
        test_target = np.array(test_dataset.targets) # (, 207, 12)
        test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor)  # (B, N, F, T)
        test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor)  # (B, N, T)
        test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle,drop_last=True)


        # Loading the graph once because it's a static graph
        for snapshot in train_dataset:
            static_edge_index = snapshot.edge_index
            break;

        p2 = time.time()
        train_t1 = time.time()
        t_mse, v_mse = train(train_loader, test_loader, batch_size, epochs, static_edge_index, device, debug=debug, mode="base")
        end = time.time()

    print(f"Runtime: {round(end - start,2)}; T-MSE: {round(t_mse, 3)}; V-MSE: {round(v_mse, 3)}")
    with open("stats.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["preprocessing","training","total", "t_mse", "v_mse"])
        writer.writerow([p2 - p1, end - train_t1, end-start, t_mse, v_mse])
if __name__ == "__main__":
    main()