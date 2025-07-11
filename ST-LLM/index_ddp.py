import torch
import numpy as np
import argparse
import time
import os
from utils import *
import random
from model_ST_LLM import ST_LLM
from ranger21 import Ranger
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from dask.distributed import LocalCluster
from dask.distributed import Client
from dask_pytorch_ddp import dispatch, results
import pandas as pd
import uuid
import json

class Args:
    def __init__(self, json_file):
        # Load the JSON file
        with open(json_file, 'r') as file:
            defaults = json.load(file)
        self.args_dict = defaults
        
        # Set attributes based on the dictionary keys and values
        for key, value in defaults.items():
            setattr(self, key, value)
    def __str__(self):
        return '\n'.join(f"{key}: {value}" for key, value in self.__dict__.items())

class trainer:
    def __init__(
        self,
        scaler,
        input_dim,
        channels,
        num_nodes,
        input_len,
        output_len,
        dropout,
        lrate,
        wdecay,
        device,
        ddp=False,
    ):
        
        self.model = ST_LLM(
            device, input_dim, channels, num_nodes, input_len, output_len, dropout
        )
        
        self.model.to(device)
        
        """
        They create their own embedding (see 'Spatial-Temporal Large Language Model for Traffic
        Prediction'), so they do not use gpt2's wte module. If not marked as unneeded, this causes a problem
        during the DDP backward pass. 
        """
        self.model.gpt.gpt2.wte.weight.requires_grad = False

        if ddp:
            self.model = DDP(self.model, gradient_as_bucket_view=True).to(device)
        
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        self.loss = MAE_torch
        self.scaler = scaler
        self.clip = 5
  

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
         
        loss.backward()
  
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = MAPE_torch(predict, real, 0.0).item()
        rmse = RMSE_torch(predict, real, 0.0).item()
        wmape = WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape
        
    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = MAPE_torch(predict, real, 0.0).item()
        rmse = RMSE_torch(predict, real, 0.0).item()
        wmape = WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True  
    torch.manual_seed(seed)


def train(start_time=None, args=None):

    dist_mode = (args.mode == "local" or args.mode == "dist")
    worker_rank = int(dist.get_rank()) if dist_mode else 0
    
    device = f"cuda:{worker_rank % 4}"
    torch.cuda.set_device(worker_rank % 4)
    
    pre_start = time.time()
    data, mean, std, x_train, x_val, x_test = index_load_dataset(args.data, args.data_key)
    pre_end = time.time()
    
    if worker_rank == 0:
        print(pre_end - pre_start, flush=True)
    
   
    train_dataset = IndexDataset(x_train,data)
    val_dataset = IndexDataset(x_val,data)

    # mirror base ST-LLM workflow that does not shuffle
    if dist_mode:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        train_loader = DataLoader(train_dataset, sampler= train_sampler, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset,sampler= val_sampler, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    scaler = StandardScaler(mean=mean, std=std)

    his_loss = []
    val_time = []
    train_time = []
    result = []

    engine = trainer(
        scaler,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
        ddp=dist_mode
    )

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []
        
        if dist_mode:
            train_sampler.set_epoch(epoch)

        t1 = time.time()
        
        for i, (x, y) in enumerate(train_loader):
            
            trainx = x.to(device).float()
            trainx = trainx.transpose(1, 3)

            """
            To mirror the standard workflow—which typically uses a standardized x dataset and an unstandardized y dataset—
            we  standardize the entire dataset and apply the inverse transform to each y-batch during evaluation.
            """
            trainy = y.to(device).float()
            trainy[...,0] = (trainy[...,0] * std) + mean
            trainy = trainy.transpose(1, 3)
            
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

            
            if args.debug and worker_rank == 0:
                print(f"Train batch: {i+1}/{len(train_loader)}", end="\r",flush=True)

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        
        if worker_rank == 0:
            print(log.format(epoch, (t2 - t1)), flush=True)
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                testx = torch.Tensor(x).to(device).float()
                testx = testx.transpose(1, 3)

                testy = torch.Tensor(y).to(device).float()
                testy[...,0] = (testy[...,0] * std) + mean
                testy = testy.transpose(1, 3)
                
                metrics = engine.eval(testx, testy[:, 0, :, :])
                
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
                valid_wmape.append(metrics[3])
                if args.debug and worker_rank == 0:
                    print(f"Val batch: {i + 1}/{len(val_loader)}", end="\r",flush=True)
        s2 = time.time()

        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        
        if worker_rank == 0:
            print(log.format(epoch, (s2 - s1)), flush=True)
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)
  
        mvalid_loss = np.sum(valid_loss)
        mvalid_mape = np.sum(valid_mape)
        mvalid_wmape = np.sum(valid_wmape)
        mvalid_rmse = np.sum(valid_rmse)
        his_loss.append(mvalid_loss)


        """
        The single-GPU workflow gets the mean loss via np.mean(loss_list), 
        where there are len(val_loader) items in loss; to mimic that, sum 
        loss across all ranks and then divide by the total number of batches.
        """
        metric_tensor = torch.tensor((mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape, len(val_loader)))
        if dist_mode:
            dist.reduce(metric_tensor, 0)
        metric_tensor = metric_tensor.clone() / metric_tensor[-1]
      
        train_m = dict(
            epoch_runtime=s2 - t1,
            train_runtime=t2 - t1,
            val_runtime=s2 - s1,
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            valid_loss=metric_tensor[0].item(),
            valid_rmse=metric_tensor[1].item(),
            valid_mape=metric_tensor[2].item(),
            valid_wmape=metric_tensor[3].item(),
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
        if worker_rank == 0:
            print(
                log.format(epoch, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape),
                flush=True,
            )
            log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
            print(
                log.format(epoch, metric_tensor[0].item(),  metric_tensor[1].item(),  metric_tensor[2].item(),  metric_tensor[3].item()),
                flush=True,
            )

    end_time = time.time()
    train_csv = pd.DataFrame(result)
    train_csv.round(8).to_csv(f"train.csv")
    if worker_rank == 0:
        with open("stats.txt", "a") as file:
                        file.write(f"total_time: {end_time - start_time}\n")
                        file.write(f"preprocess_time: {pre_end - pre_start}\n")
                        file.write(f"training_time: {end_time - train_start}\n")
                        file.write(f"train_opt_loss: {train_csv['train_loss'].min()}\n")
                        file.write(f"train_opt_rmse: {train_csv['train_rmse'].min()}\n")
                        file.write(f"train_opt_mape: {train_csv['train_mape'].min()}\n")
                        file.write(f"train_opt_wmape: {train_csv['train_wmape'].min()}\n")
                        
                        file.write(f"valid_opt_loss: {train_csv['valid_loss'].min()}\n")
                        file.write(f"valid_opt_rmse: {train_csv['valid_rmse'].min()}\n")
                        file.write(f"valid_opt_mape: {train_csv['valid_mape'].min()}\n")
                        file.write(f"valid_opt_wmape: {train_csv['valid_wmape'].min()}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the config JSON file")
    filepath = parser.parse_args().config_file
    args = Args(filepath)
    
    seed_it(6666)
    start_time = time.time()
    
    if args.data == "bike_drop":
        args.data = "data//" + args.data
        args.num_nodes = 250
    
    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        args.num_nodes = 250
    
    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        args.num_nodes = 266

    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        args.num_nodes = 266
    
    elif args.data == 'metr-la':
        args.data = "data/" + args.data + ".h5"
        args.num_nodes = 207
    elif args.data == 'pems-bay':
        args.data = "./data/" + args.data  + ".h5"
        args.num_nodes = 325
    
    if args.mode == 'local':
            cluster = LocalCluster(n_workers=args.npar)
            client = Client(cluster)
    elif args.mode == 'dist':
        client = Client(scheduler_file = f"cluster.info")
    elif args.mode == "index":
        pass  
    else:
        print(f"{args.mode} is not a valid mode; Please enter mode as 'index', 'local', or 'dist'")
        exit()
    
    if args.mode == "dist":
            for f in ['utils.py', 'model_ST_LLM.py', 'ranger21.py', 'index_ddp.py']:
                print("Uploading ", f, flush=True)
                client.upload_file(f)
    if args.mode == "local" or args.mode == "dist":
        futures = dispatch.run(client, train,
                                    start_time=start_time, args=args,
                                    backend="gloo")
        key = uuid.uuid4().hex
        rh = results.DaskResultsHandler(key)
        rh.process_results(".", futures, raise_errors=False)
        client.shutdown()
    else:
         train(start_time=start_time, args=args)
    
if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
