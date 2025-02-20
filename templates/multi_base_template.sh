#!/bin/bash
# Same as base_template.sh, but we will run a different python command for each combination of 'models_nickname' and
# 'seeds_models'. This can be useful for example if we want to run several experiments in parallel without relying on
# internal parallelization mechanisms of the experiment itself. A notable example is when using slurm where we could
# run this script as a job via sbatch and then use srun (before the python command in this script) to run the
# python command as a job_step.
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
  python $experiment_python_location $args_str $string_combination
done
