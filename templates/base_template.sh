#!/bin/bash
# This is a template to run base_experiment with all the possible arguments directly from the command line
# You can copy this script and modify the values of the arguments to run your own experiments
# Besides, you can add more arguments to the dictionaries args_dict and bool_args_dict if needed (for example when
# creating other experiments derived from base_experiment) take a look at hpo_template.sb for an example of how to add
# more arguments.
# Finally, if you need to run one python command for each combination of arguments, we have also a script for that
# called multi_base_template.sh that can be used as a template for that purpose.
environment_name=""
experiment_python_location=""

# Create a dictionary with argument names and values
declare -A args_dict=(
# base
["experiment_name"]=""
["models_nickname"]=""
["seeds_models"]=""
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

# Construct the argument string
args_str=""
for key in "${!args_dict[@]}"; do
  if [ -n "${args_dict[$key]}" ]; then
    args_str="$args_str --$key ${args_dict[$key]}"
  fi
done

for key in "${!bool_args_dict[@]}"; do
  if [ "${bool_args_dict[$key]}" -eq 1 ]; then
    args_str="$args_str --$key"
  fi
done

# Activate the conda environment and run the experiment
eval "$(conda shell.bash hook)"
conda activate $environment_name
python $experiment_python_location $args_str
