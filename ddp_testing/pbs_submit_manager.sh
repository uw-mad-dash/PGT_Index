
# compute nodes
nodes=(1 2 4 8 16 32)

# gpus per compute node
gpus=(4)

# modes to iterate over
modes=("index" "baseline")

cwd="$(pwd)"

for m in "${modes[@]}"
do
    for num_nodes in "${nodes[@]}"
    do
        for g in "${gpus[@]}"
        do
            
            echo $m,$num_nodes,$g
            
            total_nodes=$((num_nodes + 1))
            DATE=$(date +"%Y-%m-%d_%T")
            target_file="${m}_${num_nodes}_nodes.sh"
            
            # Setup PBS headers
            echo "#!/bin/bash" >> $target_file
            echo "#PBS -l select=${total_nodes}:system=polaris" >> $target_file
            echo "#PBS -l place=scatter" >> $target_file
            
            # Change to fit your needs
            echo "#PBS -l walltime=08:00:00" >> $target_file
            echo "#PBS -l filesystems=home:eagle" >> $target_file
            echo "#PBS -q queue" >> $target_file
            echo "#PBS -A project" >> $target_file
            
            echo "#PBS -o train.out" >> $target_file
            echo "#PBS -e train.err" >> $target_file
            echo "" >> $target_file

            # echo in needed variables
            echo "nodes=${num_nodes}" >> $target_file
            echo "gpus=${g}" >> $target_file
            echo "mode=${m}" >> $target_file
            dir="pems_${m}_node_${num_nodes}_${DATE}"
            echo $dir
            echo "myDIR=${cwd}/${dir}" >> $target_file
            
            # copy over needed files
            cat base_submit.sh >> $target_file
            mkdir -p $dir
            cp *.py $dir/
            mv $target_file $dir
            cp -r data/ $dir/
            cd $dir

            qsub $target_file
            sleep 2
            cd .. 
            exit
        done 
    done
done