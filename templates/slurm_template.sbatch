#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH -c N_CORES
#SBATCH -w NODE_NAME
#SBATCH --exclude=NODE_NAME1,NODE_NAME2...
#SBATCH --output=/path/to/output_dir/%x.%J.out
#SBATCH --error=/path/to/output_dir/%x.%J.err
#SBATCH --time=364-23:59:59
#SBATCH -G N_GPUS -> delete if not using GPUs, use either this line or the next one
#SBATCH --gres=mps:%OF_GPUS -> delete if not using GPUs, use either this line or the previous one
#SBATCH --array=START-END:STEP%N_SIMULTANEOUS -> delete if not using array jobs

# if we want to run multiple array jobs and overcome the limitation of slurm on the maximum number of jobs
array_start=0
array_end=500
array_step=1
array_simultaneous=50
array_batch=50
script_path="path/to/sbatch/script"

# then copy another template or modify this one accordingly
environment_name=""
experiment_python_location=""

# Create a dictionary with argument names and values
declare -A args_dict=(
# base
["experiment_name"]=""
["n_jobs"]=""
["models_params"]=""
["fits_params"]=""
["error_score"]=""
["timeout_fit"]=""
["timeout_combination"]=""
["log_dir"]=""
["log_file_name"]=""
["work_root_dir"]=""
["save_root_dir"]=""
["mlflow_tracking_uri"]=""
["dask_cluster_type"]=""
["n_workers"]=""
["n_cores_per_worker"]=""
["n_processes_per_worker"]=""
["n_threads_per_worker"]=""
["n_cores_per_task"]=""
["n_processes_per_task"]=""
["n_threads_per_task"]=""
["dask_memory"]=""
["dask_job_extra_directives"]=""
["dask_address"]=""
["n_gpus_per_worker"]=""
["n_gpus_per_task"]=""
)

declare -A bool_args_dict=(
# base
["create_validation_set"]=0
["do_not_clean_work_dir"]=0
["do_not_log_to_mlflow"]=0
["do_not_check_if_exists"]=0
["do_not_retry_on_oom"]=0
["raise_on_fit_error"]=0
)

declare -A array_args_dict=(
# Note that bash does not allow arrays inside dictionaries, so we will use strings with ',' as separators and '-' when
# we want multiple values for the same argument (basically they will be replaced by ' ')
# base
["models_nickname"]="Model1,Model2"
["seeds_models"]="0-1,2-3,4"
# in this example we are expecting the combinations Model1+0-1, Model1+2-3, Model1+4, Model2+0-1, Model2+2-3, Model2+4
)

# bash does not necessarily keep the order of the keys in the dictionary, so we will specify the order here
declare -a array_args_dict_order=("models_nickname" "seeds_models")

# Construct the argument string
args_str=""
for key in "${!args_dict[@]}"; do
  if [ -n "${args_dict[$key]}" ]; then
    args_str="$args_str --$key ${args_dict[$key]}"
  fi
done

# Add arguments strings that are boolean
for key in "${!bool_args_dict[@]}"; do
  if [ "${bool_args_dict[$key]}" -eq 1 ]; then
    args_str="$args_str --$key"
  fi
done

# Construct the cartesian product of the arrays
# the idea is to create a string like {Model1,Model2}+{0-1,2-3,4} and then evaluate it to get the cartesian product
# using bash's brace expansion
string_for_cartesian_product=""
for key in "${array_args_dict_order[@]}"; do
  str_array=${array_args_dict[$key]}
  n_elements=$(echo $str_array | tr ',' ' ' | wc -w)
  if [ $n_elements -eq 0 ]; then
    continue
  elif [ $n_elements -eq 1 ]; then
    string_for_cartesian_product="$string_for_cartesian_product+$str_array"
  else
    string_for_cartesian_product="$string_for_cartesian_product+{$str_array}"
  fi
done

# Remove the first '+' character
string_for_cartesian_product=${string_for_cartesian_product:1}

# Evaluate the string to get the cartesian product
cartesian_product=$(eval echo $string_for_cartesian_product)

# Split the string into an array (1 combination per element)
IFS=' ' read -r -a cartesian_product <<< "$cartesian_product"
# cartesian_product is now an array like ["Model1+0-1", "Model1+2-3", "Model1+4", "Model2+0-1", "Model2+2-3", "Model2+4"]

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate $environment_name

# Run one python command for each combination in the cartesian product with srun
#for i_combination in "${!cartesian_product[@]}"; do
#  string_combination=""
#  # split the string into an array
#  IFS='+' read -r -a combination <<< "${cartesian_product[$i_combination]}"
#  i_arg_name=0
#  for key in "${array_args_dict_order[@]}"; do
#    value=${combination[$i_arg_name]//[-]/ }  # replace '-' by ' '
#    string_combination="${string_combination} --${key} ${value}"
#    i_arg_name=$((i_arg_name+1))
#  done
#  # string_combination is now like "--models_nickname Model1 --seeds_models 0 1"
#  # Run the python command
#  #  python $experiment_python_location $args_str $string_combination
#  # or run it with srun to run one step
#  # exclusive to avoid conflicts with other job steps and -n 1 to run only one task with the specified number of cores
#  # note the & at the end to run the command in the background and allow parallel execution of multiple combinations
#  srun --exclusive -n 1 -c $SLURM_CPUS_PER_TASK python $experiment_python_location $args_str $string_combination &
#done
#wait

# alternatively we could also run an array job and execute the combination given by the SLURM_ARRAY_TASK_ID
i_combination=$SLURM_ARRAY_TASK_ID
if [ "$i_combination" -ge "${#cartesian_product[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID is greater than the number of combinations"
  exit 1
fi
string_combination=""
IFS='+' read -r -a combination <<< "${cartesian_product[$i_combination]}"
i_arg_name=0
for key in "${array_args_dict_order[@]}"; do
  value=${combination[$i_arg_name]//[-]/ }  # replace '-' by ' '
  string_combination="$string_combination --$key ${value}"
  i_arg_name=$((i_arg_name+1))
done
echo "Running the following comand: python $experiment_python_location $args_str $string_combination"
srun --exclusive -n 1 -c $SLURM_CPUS_PER_TASK python $experiment_python_location $args_str $string_combination

# If we are at the last array task, submit the next array job
if [ "$SLURM_ARRAY_TASK_ID" -eq "$SLURM_ARRAY_TASK_MAX" ]; then
  next_array_start=$(($SLURM_ARRAY_TASK_MIN+array_batch))
  # next_array_end will be the minimum between array_end and next_array_start+array_batch
  next_array_end=$(( array_end < next_array_start+array_batch ? array_end : next_array_start+array_batch ))
  # if next_array_start <= array_end, then we submit job, otherwise we are done
  if [ "$next_array_start" -le "$next_array_end" ]; then
    sbatch --array=$next_array_start-$next_array_end:$array_step%$array_simultaneous $script_path
  fi
fi