


cd $myDIR

if [ "$mode" = "baseline" ]; then
    echo "Running baseline DDP mode"
elif [ "$mode" = "index" ]; then
    echo "Running distributed-index-batching"
elif [ "$mode" = "batch-index" ]; then
    echo "Running generalized-distributed-index-batching"
elif [ "$mode" = "batch-baseline" ]; then
    echo "generalized batch shuffle DDP"
else
    echo "Unknown mode. Valid options are 'baseline', 'index', 'batch-index', and 'batch-baseline'"
    exit 1
fi



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


if [ "$mode" = "baseline" ]; then
    # baseline DDP
    mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node `which python3` opt_baseline.py --mode dask -np $gpus --dataset pems --dist True  &

elif [ "$mode" = "index" ]; then
    # distributed-index-batching
    mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node `which python3` pems_ddp.py --mode index -np $gpus -g true --dataset pems --dist True  &

elif [ "$mode" = "batch-index" ]; then
    # generalized-distributed-index-batching
    mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node `which python3` opt_pems_ddp.py --mode dask-index -np $gpus --dataset pems --dist True  &

elif [ "$mode" = "batch-baseline" ]; then
    # batch shuffling baseline DDP
    mpiexec -n 1 --ppn 1 --cpu-bind none --hosts $scheduler_node `which python3` opt_pems_ddp.py --mode dask -np $gpus --dataset pems --dist True  &

else
    echo "Unknown mode. Valid options are 'baseline', 'index', 'batch-index', and 'batch-baseline'"
    exit 1
fi








wait