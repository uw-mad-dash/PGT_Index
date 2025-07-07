

nodes=4
gpus=4

total=$((gpus * nodes))

readarray -t all_nodes < "$NODEFILE"

# use first node for dask scheduler and client
scheduler_node=${all_nodes[0]}
monitor_node=${all_nodes[1]}

# all nodes but first for workers
tail -n +2 $NODEFILE > worker_nodefile.txt

echo "Launching scheduler"
mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node dask scheduler --scheduler-file cluster.info &
scheduler_pid=$!

# wait for the scheduler to generate the cluster config file
while ! [ -f cluster.info ]; do
    sleep 1
    echo .
done

# Optional profiling for node 1
mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $monitor_node python3 worker_monitor.py &



echo "$total workers launching" 
mpiexec -n $total --ppn $gpus --cpu-bind none --hostfile worker_nodefile.txt dask worker --local-directory /local/scratch --scheduler-file cluster.info --nthreads 8 --memory-limit 512GB &


# **************************** Client Options ****************************
echo "Launching client"


# baseline DDP
mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node `which python3` opt_baseline.py --mode dask -np $gpus --dataset pems --dist True  &

# distributed-index-batching
# mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node `which python3` pems_ddp.py --mode index -np $gpus -g true --dataset pems --dist True  &

# generalized-distributed-index-batching
# mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node `which python3` opt_pems_ddp.py --mode dask-index -np $gpus --dataset pems --dist True  &


# batch shuffling or baseline DDP
# mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node `which python3` opt_pems_ddp.py --mode dask -np $gpus --dataset pems --dist True  &


wait