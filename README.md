
This repository contains the artifacts associated with the 2025 SC paper "PGT-I: Scaling Spatiotemporal GNNs with Memory-Efficient Distributed Training". The primary purpose of this repository is archival and to consolidate all utilized code in a single location. We integrated a cleaned, refactored, and thoroughly tested version of our methods into the [PyTorch Geoemtric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) library, and we recommend using index-batching via the official PGT repository.  

--------------------------------------------------------------------------------
This repository is an extension of [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) (PGT) designed specifically for memory efficiency, scalability, and distributed training. In particular, we make the following high-level open-source software contributions:

* Intergrated batching and sequence to sequence prediction into PGT-DCRNN
* Implemented Dask distributed data parallel DCRNN training
* Implemented Index-batching - a novel technique that significantly reduces data memory consumption and enables training on previously intratable datasets
* Implemented GPU-Index-batching: performs ST-GNN data preprocessing and training entirely in GPU memory
* Implemented Dask and index preprocessing scripts for all tested datasets
* Implemented Distributed-Index-batching: combines DDP with GPU-index-batching
* Implemented generalized-distributed-index-batching - a version of index-batching that partitions data across workers and uses batch-level shuffling to reduce communication. 






--------------------------------------------------------------------------------
**Singe Node Testing**

Examples used for testing our methods in a single-GPU context can be found in the `single_gpu_testing` folder. To recreate our experiments, begin by `cd single_gpu_testing/`


To run single GPU experiments, run the script with the dataset in its name (e.g. `python3 chicken_pox_main.py`). Each script uses the following command-line arguments:


| Argument            | Short Form | Type    | Default  | Description                                                 |
|---------------------|------------|---------|----------|-------------------------------------------------------------|
| `--mode`            | `-m`       | `str`   | `base`   | Specifies which version of the program to run. Supported options are 'base' and 'index'.            |
| `--gpu`             | `-g`       | `str`   | `False`  | Indicates whether data should be preprocessed and migrated directly to the GPU. |
| `--debug`           | `-d`       | `str`   | `False`  | Enables or disables debugging information.                 |

Once training completes, there will be three files in the current directory:
* stats.csv: summary statistics about the overall training such as minimum validation MAE, overall runtime, maximum system memory usage, etc.
* per_epoch_stats.csv: each epochs runtime, training MAE, and validation MAE.
* system_stats.csv: per-second system and GPU memory usage.

**Note that `metr_la_main.py` uses A3T-GCN rather than DCRNN.** 

--------------------------------------------------------------------------------

**PGT DDP and Multi-Node Testing**

Our DDP scripts can be found in `ddp_testing`. By default, the scripts will run on a single machine using a [Dask local cluster](https://docs.dask.org/en/stable/deploying-python.html#localcluster), allowing the user to select the number of GPUs to use. We also performed testing on SUPERCOMPUTER with up to 32 nodes (128 GPUs) using Dask's command line interface. We include an example submit script -- `submit.sh` for future use. By changing the `nodes` variable, you can change the number of workers and by changing `gpus` the number of GPUs per node. We profiled our code on a node-wide level using `worker_monitor.py`, but its launch command is commented out in `submit.sh`.


To run Dask distributed data parallel training, use the following command: `python3 pems_ddp.py <options>`. This script supports the following command-line arguments: 


| Argument             | Short Form | Type    | Default    | Description                                                              |
|----------------------|------------|---------|------------|--------------------------------------------------------------------------|
| `--mode`             | `-m`       | `str`   | `index`    | Specifies which version of the program to run. Valid options include 'dask', 'index', and 'dask-index'.                     |
| `--gpu`              | `-g`       | `str`   | `False`    | Indicates whether data should be preprocessed and migrated directly to the GPU. |
| `--debug`            | `-d`       | `str`   | `False`    | Enables or disables debugging information.                              |
| `--dist`             |            | `str`   | `False`    | Specifies if computation is distributed across multiple nodes. To do so, we leverage Dask  command line tools to spawn the following Dask constructs: scheduler, client, and workers. For an example, see `submit.sh` and for more details, see  [Dask's Documentation](https://docs.dask.org/en/latest/deploying-cli.html)      |
| `--npar`             | `-np`      | `int`   | `1`        | The number of GPUs or workers per node.                                 |
| `--dataset`          |            | `str`   | `pems-bay` | Specifies the dataset to use. Valid options include 'pems-bay', 'pems-all-la', and 'pems'.                                |


To run the baseline distributed data parallel training, use the following command: 


`python3 opt_basline.py <options>`. This script supports the following command-line arguments: 

| Argument             | Short Form | Type    | Default    | Description                                                              |
|----------------------|------------|---------|------------|--------------------------------------------------------------------------|
| `--mode`             | `-m`       | `str`   | `dask`    | Specifies which version of the program to run. Valid options include 'dask'                     |
| `--debug`            | `-d`       | `str`   | `False`    | Enables or disables debugging information.                              |
| `--dist`             |            | `str`   | `False`    | Specifies if computation is distributed across multiple nodes. To do so, we leverage Dask  command line tools to spawn the following Dask constructs: scheduler, client, and workers. For an example, see `submit.sh` and for more details, see  [Dask's Documentation](https://docs.dask.org/en/latest/deploying-cli.html)      |
| `--npar`             | `-np`      | `int`   | `1`        | The number of GPUs or workers per node.                                 |
| `--dataset`          |            | `str`   | `pems-bay` | Specifies the dataset to use. Valid options include 'pems-bay', 'pems-all-la', and 'pems'.                                |
 


To run the further optimized Dask distributed data parallel training with batch-level shuffling, use the following command: 

`python3 opt_pems_ddp.py <options>`. This script supports the following command-line arguments: 
| Argument             | Short Form | Type    | Default    | Description                                                              |
|----------------------|------------|---------|------------|--------------------------------------------------------------------------|
| `--mode`             | `-m`       | `str`   | `dask`    | Specifies which version of the program to run. Valid options include 'dask' and 'dask-index'                    |
| `--debug`            | `-d`       | `str`   | `False`    | Enables or disables debugging information.                              |
| `--dist`             |            | `str`   | `False`    | Specifies if computation is distributed across multiple nodes. To do so, we leverage Dask  command line tools to spawn the following Dask constructs: scheduler, client, and workers. For an example, see `submit.sh` and for more details, see  [Dask's Documentation](https://docs.dask.org/en/latest/deploying-cli.html)      |
| `--npar`             | `-np`      | `int`   | `1`        | The number of GPUs or workers per node.                                 |
| `--dataset`          |            | `str`   | `pems-bay` | Specifies the dataset to use. Valid options include 'pems-bay', 'pems-all-la', and 'pems'.                                |

--------------------------------------------------------------------------------
**ST-LLM**

To demostrate distributed-index-batching's broader applicability, we intergrated it into [ST-LLM](https://github.com/ChenxiLiu-HNU/ST-LLM/tree/062c7ada936ea89986bd952be18d6ecd198953b3) from [Spatial-Temporal Large Language Model for Traffic Prediction](https://arxiv.org/abs/2401.10134). The code is contained within TODO. Note that a older commit of ST-LLM is used as that is when the version of the model we intergrated index-batching with. 
---------------------------------------------------------
**Datasets**

We tested and compared model performance with the following PyTorch Geometric Temporal datasets to establish that index-batching has no negative impact on accurary and reduces memory footprint:

* Hungary Chickenpox 
* PeMS-BAY
* Windmill-Large
* Metr-LA

And we employed the following two larger datasets to test single-GPU scalability and multi-GPU, multi-node distributed DDP training scalability:
* PeMS-All-LA
* The full PeMS dataset


We could not download the windmill-large dataset using the existing `json.loads(urllib.request.urlopen(url).read().decode())` contained within the PGT windmilllarge dataloader. Instead, we manually downloaded the file from [graphmining.ai](https://graphmining.ai/temporal_datasets/windmill_output.json) and placed it in a subfolder `data/`; as such, our code loads the data from `data/windmill_output.json`. 

For the PeMS-based datasets, we assume the following about the data files
1. They are contained within a subdirectory `data/` of the active directory
2. The data file name follows the format `<dataset>.h5` (e.g. `pems-all-la.h5`)
3. The adj matrix file name follows the format `adj_mx_<dataset>.pkl` (e.g. `adj_mx_pems-all-la.pkl`)

---------------------------------------------------------
**Installation**

Clone our repository and change directory to the downloaded repository. Then run

```sh
pip3 install .
```

[pytorch-install]: https://pytorch.org/get-started/locally/
[pyg-install]: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html



