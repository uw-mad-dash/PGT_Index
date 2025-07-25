#!/bin/bash

mkdir -p data 

python3 worker_monitor.py &
python3 pems_ddp.py -m index -np 4 -g true --dataset pems  > output.log 2>&1
mkdir PemsIndex
mv *.csv PemsIndex/
mv output.log PemsIndex/


python3 worker_monitor.py &
python3 opt_baseline.py -m dask -np 4 --dataset pems  > output.log 2>&1
mkdir PemsBase
mv *.csv PemsBase/
mv output.log PemsBase/
