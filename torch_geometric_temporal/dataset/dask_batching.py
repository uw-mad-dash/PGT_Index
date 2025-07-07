import torch
from torch.utils.data import Dataset
from dask.array.lib.stride_tricks import sliding_window_view
from dask.delayed import delayed
import dask.array as da
import dask.dataframe as dd

import pandas as pd
import numpy as np

class DaskDataset(Dataset):
    def __init__(self,x, y, lazy_batching=False):
         self.x = x 
         self.y = y
         self.lb = lazy_batching
         self.timings = []
        
    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.lb:
            
            x_tensor = torch.from_numpy(self.x[idx].compute())
            y_tensor = torch.from_numpy(self.y[idx].compute())
            
            return x_tensor, y_tensor
        
        return self.x[idx], self.y[idx]
    
    @staticmethod
    def preprocess(filepath, key, horizon=12, chunkSetting="auto", ratio = (0.7, 0.1, 0.2)):
        
        def readPD():
            df = pd.read_hdf(filepath, key=key)
            if 'bay' in filepath:
                pass
            else:
                df.index.freq='5min'  # Manually assign the index frequency
            df.index.freq = df.index.inferred_freq
            return df
        
        dfs = delayed(readPD)()
        df = dd.from_delayed(dfs)



        num_samples, num_nodes = df.shape

        num_samples = num_samples.compute()
        
        x_offsets = np.sort(np.arange(-11, 1, 1))
        y_offsets = np.sort(np.arange(1, 13, 1))
        
        
        data1 =  df.to_dask_array(lengths=True)
        data1 = da.expand_dims(data1, axis=-1)
        data1 = data1.rechunk(chunkSetting)


        data2 = da.tile((df.index.values.compute() - df.index.values.compute().astype("datetime64[D]")) / np.timedelta64(1, "D"), [1, num_nodes, 1]).transpose((2, 1, 0))
        data2 = data2.rechunk((data1.chunks))
        
        
        data_array = da.concatenate([data1, data2], axis=-1)
        data_array = data_array.rechunk(chunkSetting)


        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        total = max_t - min_t

        window_size = horizon
        original_shape = data_array.shape

        
        # Define the window shape
        window_shape = (window_size,) + original_shape[1:]  # e.g. Pems-la (12, 207, 2)

        # Use sliding_window_view to create the sliding windows
        sliding_windows = sliding_window_view(data_array, window_shape).squeeze()
        
        x_array = sliding_windows[:total]
        y_array = sliding_windows[window_size:]
        del data_array
        del sliding_windows





        num_samples = x_array.shape[0]
        num_train = round(num_samples * ratio[0])

        
        num_test = round(num_samples * ratio[2])
        num_val = num_samples - num_test - num_train

        
        # create train, validation, and test splits 
        x_train = x_array[:num_train]
        y_train = y_array[:num_train]
        
        x_val = x_array[num_train: num_train + num_val]
        y_val = y_array[num_train: num_train + num_val]

        x_test = x_array[-num_test:]
        y_test = y_array[-num_test:]
        
        mean = x_train[..., 0].mean()
        std = x_train[..., 0].std()
        

        x_train[..., 0] = (x_train[..., 0] - mean) / std
        y_train[..., 0] = (y_train[..., 0] - mean) / std
        
        x_val[..., 0] = (x_val[..., 0] - mean) / std
        y_val[..., 0] = (y_val[..., 0] - mean) / std
        
        x_test[..., 0] = (x_test[..., 0] - mean) / std
        y_test[..., 0] = (y_test[..., 0] - mean) / std

        return x_train, y_train, x_val, y_val, x_test, y_test, mean, std