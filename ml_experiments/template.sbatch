#!/bin/sh
#SBATCH --job-name=[JOB_NAME]
#SBATCH -c [N_CORES]
#SBATCH -w [CLUST]
#SBATCH --output=[OUTPUT_FILE_STR]
#SBATCH --error=[ERROR_FILE_STR]
#SBATCH --time=364-23:59:59
eval "$(conda shell.bash hook)"
conda activate [ENV]
cd [PATH]
python [SCRIPT.py] --experiment_name [EXPERIMENT_NAME] --log_dir [LOG_DIR] --work_root_dir [WORK_DIR] --save_root_dir [SAVE_DIR] --mlflow_tracking_uri [RUNNING_MLFLOW_URI] --n_jobs [N_JOBS] --models_nickname [MODELS_NICKNAMES] --seeds_models [SEEDS_MODELS] --log_file_name [LOG_FILE_NAME] --dask_cluster_type [DASK_TYPE] --dask_memory [DASK_MEMORY_STR] --n_workers [N_WORKERS] --n_cores [N_CORES]
