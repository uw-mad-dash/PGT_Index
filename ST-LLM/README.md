This repo is an extension of the model introduced in [*"Spatial-Temporal Large Language Model for Traffic Prediction"* paper](https://github.com/ChenxiLiu-HNU/ST-LLM/blob/main/ST-LLM.pdf).

[Original Repo link](https://github.com/ChenxiLiu-HNU/ST-LLM).


[Link to download](https://github.com/chnsh/DCRNN_PyTorch) the PeMS-Bay h5 and adj-matrix file. The files must be a directory named data.


Execute single-GPU index-batching
```
python3 index_ddp.py --config-file single_gpu_bay_config.json
```

Execute single-GPU index-batching
```
python3 index_ddp.py --config-file single_gpu_bay_config.json
```

Execute multi-GPU single-node index-batching
```
python3 index_ddp.py --config-file multi_gpu_bay_config.json
```

