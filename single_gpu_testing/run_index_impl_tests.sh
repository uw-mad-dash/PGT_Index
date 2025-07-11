#!/bin/bash

mkdir -p data/

echo "Running CPU index-batching"
python3 pems_main.py -m index  > output.log 2>&1
mkdir PemsIndex
mv *.csv PemsIndex/
mv output.log PemsIndex/


echo "Running GPU index-batching"
python3 pems_main.py -m index -g true > output.log 2>&1
mkdir PemsGPUIndex
mv *.csv PemsGPUIndex/
mv output.log PemsGPUIndex/