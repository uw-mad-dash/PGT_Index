#!/bin/bash

# Check that num_iters was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <num_iters>"
  exit 1
fi

# Pass number of iterations as the first argument
num_iters=$1

for ((i=1; i<=num_iters; i++))
do
    echo "Running iteration $i"


    python3 chicken_pox_main.py -m base  > output.log 2>&1
    mkdir chickenPoxBase${i}
    mv *.csv chickenPoxBase${i}/
    mv output.log chickenPoxBase${i}/

    python3 chicken_pox_main.py -m index  > output.log 2>&1
    mkdir chickenPoxIndex${i}
    mv *.csv chickenPoxIndex${i}/
    mv output.log chickenPoxIndex${i}/
    
    
    python3 pems_bay_main.py -m base  > output.log 2>&1
    mkdir PemsBayBase${i}
    mv *.csv PemsBayBase${i}/
    mv output.log PemsBayBase${i}/

    python3 pems_bay_main.py -m index  > output.log 2>&1
    mkdir PemsBayIndex${i}
    mv *.csv PemsBayIndex${i}/
    mv output.log PemsBayIndex${i}/
    

    python3 windmill_main.py -m base  > output.log 2>&1
    mkdir WindmillBase${i}
    mv *.csv WindmillBase${i}/
    mv output.log WindmillBase${i}/

    python3 windmill_main.py -m index  > output.log 2>&1
    mkdir WindmillIndex${i}
    mv *.csv WindmillIndex${i}/
    mv output.log WindmillIndex${i}/


done