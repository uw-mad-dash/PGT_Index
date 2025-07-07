

import torch
from torch.utils.data import Dataset

import dask.array as da
from dask.distributed import Client
from dask.array.lib.stride_tricks import sliding_window_view
from dask.distributed import wait as Wait
from dask.delayed import delayed
import dask.dataframe as dd

import pandas as pd
import numpy as np
import json
import urllib

class IndexDataset(Dataset):
    """
    A custom PyTorch Dataset that implements indexBatching and lazyIndexBatching.
    It also supports intergration with GPU indexPreprocessing and lazyPreprocessing.

    Args:
            indices (array-like): Indices corresponding to the time slicies.
            data (array-like or Dask array): The dataset to be indexed.
            horizon (int): The prediction period for the dataset.
            lazy (bool, optional): Whether to use Dask lazy loading (distribute the data across all workers). Defaults to False.
            gpu (bool, optional): If the data is already on the GPU. Defaults to False.
    """
    def __init__(self,indices, data, horizon, lazy=False, gpu=False):
         self.indices = indices 
         self.data = data
         self.horizon = horizon
         self.lazy = lazy
         self.gpu = gpu
        #  print("Horizion: ", horizon, flush=True)
        #  print(indices[0])
        
    def __len__(self):
        # Return the number of samples
        return self.indices.shape[0]

    def __getitem__(self, x):
        """
        Retrieve a data sample and its corresponding target based on the index.

        Args:
            x (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple (x, y), where `x` is the input sequence and `y` is the target sequence.
        """

        idx = self.indices[x]
        
        # calcuate the offset based on the horizon value
        y_start = idx + self.horizon

        # if the data is already on the gpu (likely due to using index-gpu-preprocessing), return tensor-slice
        if self.gpu:
            return self.data[idx:y_start,...], self.data[y_start:y_start + self.horizon,...]
        
        else:
            # if utilizing lazyBatching, gather the data on to this worker and convert to tensor
            if self.lazy:
                return torch.from_numpy(self.data[idx:y_start,...].compute()),torch.from_numpy(self.data[y_start:y_start + self.horizon,...].compute())
            else:
                return torch.from_numpy(self.data[idx:y_start,...]), torch.from_numpy(self.data[y_start:y_start + self.horizon,...])
    
    @staticmethod
    def preprocess(filename, horizon, dataset,
                        h5_key=None, freqIndex=None, add_time_in_day=False, add_day_in_week=False, gpu=-1, ratio=(0.7, 0.1, 0.2),
                        compare_to_pgt=False       
                ):
        """
        Performs index-preprocessing on time-series data to train spatiotemporal models.

        Args:
            filename (str): Path to the file containing the data.
            horizon (int): The prediction period (e.g. 6 time periods forward).
            dataset (str): The dataset specific preprocessing method to use.

            h5_key (str,optional): The key to access the dataset if it is an HDF5 file.
            freqIndex (str, optional): The frequency of the time index (e.g., '5min'). If provided, it manually sets the frequency.
            add_time_in_day (bool, optional): Whether to add "time of day" as a feature. Defaults to False.
            add_day_in_week (bool, optional): Whether to add "day of the week" as a feature. Defaults to False.
            gpu (int, optional): The GPU to use if GPU-specific processing is used. Also controls if the GPU is used.
            ratio ( (float, float, float), optional): The desired train, validation, and test split, respectively.
        Returns:
            tuple: A tuple containing:
                - `data` (np.ndarray or torch.tensor): The standardized data. This will be a numpy array if preprocessing is
                                                    performed on the CPU and a torch tensor if performed on the GPU.  
                - `mean` (float): The mean of the dataset as if SWA was used.
                - `std` (float): The standard deviation of the dataset as if SWA was used.
                - `x_train` (np.ndarray): Indices for the time slices in the train set.
                - `x_val` (np.ndarray): Indices for the time slices in the valdiation set.
                - `x_test` (np.ndarray): Indices for the time slices in the test set.

        Raises:
            NotImplementedError: If GPU support is requested for "day of the week" feature processing.
        """
        if "pems" in dataset:

            if dataset == "pems-bay" and compare_to_pgt:
                data = np.load(filename).transpose((1, 2, 0))
                data = data.astype(np.float32)
                if gpu != -1:
                    raise NotImplementedError("Currently GPU index preprocessing is not implemented for PGT preprocessing")
                else:
                    # Normalise as in DCRNN paper (via Z-Score Method)
                    means = np.mean(data, axis=(0, 2))
                    data = data - means.reshape(1, -1, 1)
                    stds = np.std(data, axis=(0, 2))
                    data = data / stds.reshape(1, -1, 1)
                    data = data.transpose((2, 0, 1))
                    
                num_samples = data.shape[0]
                x_i = np.arange(num_samples - (2 * horizon - 1))
                num_samples = x_i.shape[0]
                num_train = round(num_samples * ratio[0])
                num_test = round(num_samples * ratio[2])
                num_val = num_samples - num_train - num_test

                x_train = x_i[:num_train]
                x_val = x_i[num_train: num_train + num_val]
                x_test = x_i[-num_test:]
                
                return data, means, stds, x_train, x_val, x_test
            else:

                df = pd.read_hdf(filename, h5_key)
                num_samples, num_nodes = df.shape
                

                
                if freqIndex:
                    df.index.freq = freqIndex
                    df.index.freq = df.index.inferred_freq

                
                if gpu != -1:

                    features = 1 + add_time_in_day + add_day_in_week
                    data = torch.empty((num_samples, num_nodes, features), device='cuda')
                    data[...,0] = torch.tensor(df.values).to(f"cuda:{gpu}")
                    
                else:
                    data = np.expand_dims(df.values, axis=-1)
                    data_list = [data[:]]


                # add time of day / day of week info. it will be of shape (num_samples, num_nodes, 1)
                if add_time_in_day:

                    if gpu != -1:
                        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
                        data[...,1] = torch.squeeze(torch.tensor(np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)))).to(f"cuda:{gpu}")
                    else:
                        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
                        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                        data_list.append(time_in_day)
                
                if add_day_in_week:

                    # TODO add GPU support
                    if gpu:
                        raise NotImplementedError("Currently GPU index preprocessing does not support data with a day-of-week feature.")
                    else:
                        day_of_week = np.array(df.index.dayofweek)[:, np.newaxis]
                        day_in_week = np.zeros(shape=(num_samples, num_nodes, 1))
                        day_in_week[...,0] = day_of_week
                        data_list.append(day_of_week)


                if gpu == -1: data = np.concatenate(data_list, axis=-1)


                
                
                
                
                # calculate all valid windows
                x_i = np.arange(num_samples - (2 * horizon - 1))
            
                bin = IndexDataset._binCalc(horizon, x_i.shape[0], ratio=ratio[0])
                

                num_samples = x_i.shape[0]
                
                num_train = round(num_samples * ratio[0])
                cutoff = bin.shape[0]
                total_entries = num_train * horizon * num_nodes
                
                
                if gpu != -1: 
                    gpuBin = torch.tensor(bin).to(f"cuda:{gpu}")
                    
                    del bin
                    mean = (data[: cutoff,..., 0].sum(dim=1) * gpuBin).sum() / total_entries
                    std = torch.sqrt((((data[:cutoff, ..., 0] - mean) ** 2).sum(dim=1) * gpuBin).sum() / total_entries)
                    del gpuBin
                else:
                    # Manipulate the arrays to calculate mean/std as if SWA was applied
                    mean = (data[: cutoff,..., 0].sum(axis=1) * bin).sum() / total_entries
                    std = np.sqrt(  ((np.square(data[: cutoff, ..., 0] - mean)).sum(axis=1) * bin ).sum() / total_entries )
                
                # Standardize the dataset
                data[..., 0] = (data[..., 0] - mean) / std
                
                num_test = round(num_samples * ratio[2])
                num_val = num_samples - num_train - num_test
                x_train = x_i[:num_train]
                x_val = x_i[num_train:num_train + num_val ]
                x_test = x_i[-num_test:]
                
        
                return data, mean, std, x_train, x_val, x_test
        elif "windmill" in dataset.lower():
            with open(filename, "r") as file:
                dataset = json.load(file)
            
            data = np.stack(dataset["block"])
            edges = np.array(dataset["edges"]).T
            edge_weights = np.array(dataset["weights"]).T
            
            num_samples = data.shape[0]
        
            if gpu != -1:
                data = torch.tensor(data, dtype=torch.float).to(f"cuda:{gpu}")
                mean = torch.mean(data, axis=0)
                std = torch.std(data, axis=0) + 10 ** -10
                data  = (data - mean) / std
                data = data.unsqueeze(-1)
            else:
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0) + 10 ** -10
                data  = (data - mean) / std
                data = np.expand_dims(data, axis=-1)

            x_i = np.arange(num_samples - (2 * horizon - 1))
            num_samples = x_i.shape[0]
            num_train = round(num_samples * ratio[0])
            num_test = round(num_samples * ratio[2])
            num_val = num_samples - num_train - num_test

            x_train = x_i[:num_train]
            x_val = x_i[num_train: num_train + num_val]
            x_test = x_i[-num_test:]

            return data, mean, std, x_train, x_val, x_test, edges, edge_weights
        
        elif "chickenpox" in dataset.lower():
            url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"
            dataset = json.loads(urllib.request.urlopen(url).read())


            
            edges = np.array(dataset["edges"]).T
            edge_weights = np.ones(edges.shape[1])

            data = np.array(dataset["FX"])

            num_samples = data.shape[0]
            
            if gpu != -1:
                data = torch.tensor(data, dtype=torch.float).to(f"cuda:{gpu}")
                data = data.unsqueeze(-1)
            else:

                data = np.expand_dims(data, axis=-1)

  
            x_i = np.arange(num_samples - (2 * horizon - 1))
            
            num_samples = x_i.shape[0]
            num_train = round(num_samples * ratio[0])
            num_test = round(num_samples * ratio[2])
            num_val = num_samples - num_train - num_test

            x_train = x_i[:num_train]
            x_val = x_i[num_train: num_train + num_val]
            x_test = x_i[-num_test:]

            return data, x_train, x_val, x_test, edges, edge_weights


    @staticmethod
    def dask_preprocess(h5_filename, h5_key, horizon, 
                        freqIndex=None, add_time_in_day=False, add_day_in_week=False, target_chunk_size=None, ratio=(0.7, 0.1, 0.2)):
        
        def readPD():
            df = pd.read_hdf(h5_filename, key=h5_key)
            
            if freqIndex:
                df.index.inferred_freq = freqIndex
            else:
                df.index.freq = df.index.inferred_freq
            
            return df


        if target_chunk_size == None:
            target_chunk_size = "auto"
        
        dfs = delayed(readPD)()
        df = dd.from_delayed(dfs)
        num_samples, num_nodes = df.shape
        num_samples = num_samples.compute()
        
        data_list = []
        speedData =  df.to_dask_array(lengths=True)
            
        speedData = da.expand_dims(speedData, axis=-1)
        speedData = speedData.rechunk("auto")
        data_list.append(speedData)

        
        if add_time_in_day:
            timeData = da.tile((df.index.values.compute() - df.index.values.compute().astype("datetime64[D]")) / np.timedelta64(1, "D"), [1, num_nodes, 1]).transpose((2, 1, 0))
            timeData = timeData.rechunk((speedData.chunks))
            data_list.append(timeData)

        if add_day_in_week:
            raise NotImplementedError("Currently  lazyIndex preprocessing does not support data with a day-of-week feature.")

        data = da.concatenate(data_list, axis=-1)

       
        # TODO: change so ratio is not hard coded
        bin = IndexDataset._binCalc(horizon, num_samples - (2 * horizon - 1), ratio=ratio[0])


        num_samples = num_samples - (2 * horizon - 1)
        num_train = round(num_samples * ratio[0])
    
        

        
        cutoff = bin.shape[0]
        total_entries = num_train * horizon * num_nodes
        


        mean = (data[: cutoff,..., 0].sum(axis=1) * bin).sum() / total_entries
        std = da.sqrt(  ((da.square(data[: cutoff, ..., 0] - mean)).sum(axis=1) * bin ).sum() / total_entries )
        

        data[..., 0] = (data[..., 0] - mean) / std
        
        data = data.rechunk(target_chunk_size)

        return data, mean, std
        
        

    
    @staticmethod
    def _binCalc(horizon, data_length, ratio=0.7):
        """
        Calcuates how many times a given element would have appeared if 
        sliding window analysis (SWA) was applied to the input data. These values are used 
        to calcuate the mean and std, allowing dataset standardization. 
        """
        # calcuate the length of train set as if we applied SWA
        bin_len = round(data_length * ratio) + (horizon - 1)
        
        # Step 1: pre-alloc
        bin = np.empty(bin_len, dtype=int)
        
        # Step 2: Fill in the ascending part [1, 2, ..., horizon]
        bin[:horizon] = np.arange(1, horizon + 1)
        
        # Step 3: Calculate how many times to repeat horizon
        remaining_elements = bin_len - (horizon * 2)
        bin[horizon:horizon+remaining_elements+1] = horizon
        
        # Step 4: Fill in the descending part [horizon - 1, ..., 1]
        bin[horizon + remaining_elements + 1:] = np.arange(horizon - 1, 0, -1)

        return bin