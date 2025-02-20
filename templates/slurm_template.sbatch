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
["n_cores"]=""
["n_processes"]=""
["dask_memory"]=""
["dask_job_extra_directives"]=""
["dask_address"]=""
["n_gpus"]=""
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
# Note that bash does not allow arrays inside dictionaries, so we will use strings
# base
["models_nickname"]="Model1 Model2"
["seeds_models"]="0 1"
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
# the idea is to create a string like {Model1,Model2}-{0,1} and then evaluate it to get the cartesian product
# using bash's brace expansion
string_for_cartesian_product=""
for key in "${array_args_dict_order[@]}"; do
  str_array=${array_args_dict[$key]}
  n_elements=$(echo $str_array | wc -w)
  str_array=$(echo $str_array | tr ' ' ',')
  if [ $n_elements -eq 0 ]; then
    continue
  elif [ $n_elements -eq 1 ]; then
    string_for_cartesian_product="$string_for_cartesian_product-$str_array"
  else
    string_for_cartesian_product="$string_for_cartesian_product-{$str_array}"
  fi
done

# Remove the first '-' character
string_for_cartesian_product=${string_for_cartesian_product:1}

# Evaluate the string to get the cartesian product
cartesian_product=$(eval echo $string_for_cartesian_product)

# Split the string into an array (1 combination per element)
IFS=' ' read -r -a cartesian_product <<< "$cartesian_product"
# cartesian_product is now an array like ["Model1-0" "Model1-1" "Model2-0" "Model2-1"]

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate $environment_name

# Run one python command for each combination in the cartesian product
for i_combination in "${!cartesian_product[@]}"; do
  string_combination=""
  # split the string into an array
  IFS='-' read -r -a combination <<< "${cartesian_product[$i_combination]}"
  i_arg_name=0
  for key in "${array_args_dict_order[@]}"; do
    string_combination="$string_combination --$key ${combination[$i_arg_name]}"
    i_arg_name=$((i_arg_name+1))
  done
  # string_combination is now like "--models_nickname Model1 --seeds_models 0"
  # Run the python command
  #  python $experiment_python_location $args_str $string_combination
  # or run it with srun to run one step
  # exclusive to avoid conflicts with other job steps and -n 1 to run only one task with the specified number of cores
  # note the & at the end to run the command in the background and allow parallel execution of multiple combinations
  srun --exclusive -n 1 -c $SLURM_CPUS_PER_TASK python $experiment_python_location $args_str $string_combination &
done
wait

# alternatively we could also run an array job and execute the combination given by the SLURM_ARRAY_TASK_ID
#i_combination=$SLURM_ARRAY_TASK_ID
#if [ "$i_combination" -ge "${#cartesian_product[@]}" ]; then
#  echo "SLURM_ARRAY_TASK_ID is greater than the number of combinations"
#  exit 1
#fi
#string_combination=""
#IFS='-' read -r -a combination <<< "${cartesian_product[$i_combination]}"
#i_arg_name=0
#for key in "${array_args_dict_order[@]}"; do
#  string_combination="$string_combination --$key ${combination[$i_arg_name]}"
#  i_arg_name=$((i_arg_name+1))
#done
#srun --exclusive -n 1 -c $SLURM_CPUS_PER_TASK python $experiment_python_location $args_str $string_combination
