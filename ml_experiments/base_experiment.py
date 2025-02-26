from __future__ import annotations
import argparse
import shlex
import tempfile
import time
from multiprocessing import cpu_count
from pathlib import Path
from shutil import rmtree, copytree
from typing import Optional
import mlflow
import os
import logging
import warnings
import numpy as np
from distributed import WorkerPlugin, Worker, Client
import dask
from dask.distributed import LocalCluster, get_worker, as_completed
from dask_jobqueue import SLURMCluster
from tqdm.auto import tqdm
from resource import getrusage, RUSAGE_SELF
import json
from abc import ABC, abstractmethod
from ml_experiments.utils import flatten_dict, get_git_revision_hash, set_mlflow_tracking_uri_check_if_exists
from func_timeout import func_timeout, FunctionTimedOut
try:
    import torch
    torch_available = True
except ImportError:
    torch = None
    torch_available = False

warnings.simplefilter(action='ignore', category=FutureWarning)


class MLFlowCleanupPlugin(WorkerPlugin):
    def teardown(self, worker: Worker):
        if mlflow.active_run() is not None:
            mlflow.log_param('EXCEPTION', f'KILLED, worker status {worker.status}')
            mlflow.end_run('KILLED')


class LoggingSetterPlugin(WorkerPlugin):
    def __init__(self, logging_config=None):
        self.logging_config = logging_config if logging_config is not None else {}
        super().__init__()

    def setup(self, worker: Worker):
        logging.basicConfig(**self.logging_config)


def log_and_print_msg(first_line, verbose, verbose_level, **kwargs):
    if verbose >= verbose_level:
        slurm_job_id = os.getenv('SLURM_JOB_ID', None)
        if slurm_job_id is not None:
            first_line = f"SLURM_JOB_ID: {slurm_job_id}\n{first_line}"
        first_line = f"{first_line}\n"
        first_line += "".join([f"{key}: {value}\n" for key, value in kwargs.items()])
        print(first_line)
        logging.info(first_line)


class BaseExperiment(ABC):
    def __init__(
            self,
            # model specific
            models_nickname: Optional[list[str]] = None,
            seeds_models: Optional[list[int]] = None,
            n_jobs: int = 1,
            models_params: Optional[dict] = None,
            fits_params: Optional[dict] = None,
            # parameters of experiment
            experiment_name: str = 'base_experiment',
            create_validation_set: bool = False,
            log_dir: str | Path = Path.cwd() / 'logs',
            log_file_name: Optional[str] = None,
            work_root_dir: str | Path = Path.cwd() / 'work',
            save_root_dir: Optional[str | Path] = None,
            clean_work_dir: bool = True,
            raise_on_fit_error: bool = False,
            parser: Optional = None,
            error_score: str = 'raise',
            timeout_fit: Optional[int] = None,
            timeout_combination: Optional[int] = None,
            verbose: int = 1,
            # mlflow specific
            log_to_mlflow: bool = True,
            mlflow_tracking_uri: str = 'sqlite:///' + str(Path.cwd().resolve()) + '/ml_experiments.db',
            check_if_exists: bool = True,
            # parallelization
            dask_cluster_type: Optional[str] = None,
            n_workers: int = 1,
            n_processes_per_worker: int = 1,
            n_cores_per_worker: int = 1,
            n_threads_per_worker: int = 2,
            n_processes_per_task: int = 0,
            n_cores_per_task: int = 0,
            n_threads_per_task: Optional[int] = None,
            dask_memory: Optional[str] = None,
            dask_job_extra_directives: Optional[str] = None,
            dask_address: Optional[str] = None,
            # gpu specific
            n_gpus_per_worker: float = 0.0,
            n_gpus_per_task: Optional[float] = None,
    ):
        """Base experiment.

        This class allows to perform experiments for machine learning models. It is a base class that can be inherited
        to perform specific experiments. It allows to perform experiments with different models, datasets and
        resampling strategies. It also allows to log the results to mlflow and to parallelize the experiments with
        dask. We can also run a single experiment with the run_* meth

        Parameters
        ----------
        models_nickname :
            The nickname of the models to be used in the experiment. The nickname must be one of the keys of the
            models_dict.
        seeds_models :
            The seeds to be used in the models.
        n_jobs :
            Number of threads/cores to be used by the model if it supports it. Defaults to 1.
        models_params :
            Dictionary with the parameters of the models. The keys must be the nickname of the model and the values
            must be a dictionary with the parameters of the model. In case only one dictionary is passed, it will be
            used for all models. Defaults to None.
        fits_params :
            Dictionary with the parameters of the fits. The keys must be the nickname of the model and the values
            must be a dictionary with the parameters of the fit. In case only one dictionary is passed, it will be
            used for all models. Defaults to None.
        experiment_name :
            The name of the experiment. Defaults to 'base_experiment'.
        create_validation_set :
            If True, create a validation set. Defaults to False.
        log_dir :
            The directory where the logs will be saved. Defaults to 'logs'.
        log_file_name :
            The name of the log file. If None, it will be the experiment_name. Defaults to None.
        work_root_dir :
            The directory where the intermediate outputs will be saved. Defaults to 'work'.
        save_root_dir :
            The directory where the final trained models will be saved.
        clean_work_dir :
            If True, clean the work directory after running the experiment. Defaults to True.
        raise_on_fit_error :
            If True, raise an error if it is encountered when fitting the model. Defaults to False.
        parser :
            The parser to be used in the experiment. Defaults to None, which means that the parser will be created.
        error_score :
            The default value to be used if a error occurs when evaluating the model. Defaults to 'raise' which
            raises an error.
        log_to_mlflow :
            If True, log the results to mlflow. Defaults to True.
        mlflow_tracking_uri :
            The uri of the mlflow server. Defaults to 'sqlite:///' + str(Path.cwd().resolve()) + '/ml_experiments.db'.
        check_if_exists :
            If True, check if the experiment already exists in mlflow. Defaults to True.
        dask_cluster_type :
            The type of the dask cluster to be used. It can be 'local' or 'slurm'. Defaults to None, which means
            that dask will not be used.
        n_workers :
            The number of workers to be used in the dask cluster. Defaults to 1.
        n_processes_per_worker :
            The number of processes to be used in the dask cluster. Defaults to 1.
        n_cores_per_worker :
            The number of cores to be used in the dask cluster. Defaults to 1.
        n_threads_per_worker :
            The number of threads to be used in the dask cluster. Defaults to 2.
        n_processes_per_task :
            The number of processes per task. Defaults to 0.
        n_cores_per_task :
            The number of cores per task. Defaults to 0.
        n_threads_per_task:
            The number of threads per task. Defaults to None which sets the number of threads per task to be the same
            as n_jobs.
        dask_memory :
            The memory to be used in the dask cluster. Defaults to None.
        dask_job_extra_directives :
            The extra directives to be used in the dask cluster. Defaults to None.
        dask_address :
            The address of an initialized dask cluster. Defaults to None.
        n_gpus_per_worker :
            The number of gpus per worker. Defaults to 0.
        n_gpus_per_task :
            The number of gpus per task. Defaults to None which uses the same value as n_gpus_per_worker.
        """
        self.models_nickname = models_nickname if models_nickname else []
        self.seeds_models = seeds_models if seeds_models else [0]
        self.n_jobs = n_jobs

        self.models_params = self._validate_dict_of_models_params(models_params, self.models_nickname)
        self.fits_params = self._validate_dict_of_models_params(fits_params, self.models_nickname)

        # parallelization
        self.dask_cluster_type = dask_cluster_type
        self.n_workers = n_workers
        self.n_cores_per_worker = n_cores_per_worker
        self.n_processes_per_worker = n_processes_per_worker
        self.n_threads_per_worker = n_threads_per_worker
        self.n_processes_per_task = n_processes_per_task
        self.n_cores_per_task = n_cores_per_task
        self._n_threads_per_task = n_threads_per_task
        self.dask_memory = dask_memory
        self.dask_job_extra_directives = dask_job_extra_directives
        self.dask_address = dask_address
        self.n_gpus_per_worker = n_gpus_per_worker
        self._n_gpus_per_task = n_gpus_per_task

        self.experiment_name = experiment_name
        self.create_validation_set = create_validation_set
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        if isinstance(work_root_dir, str):
            work_root_dir = Path(work_root_dir)
        self.work_root_dir = work_root_dir
        if isinstance(save_root_dir, str):
            save_root_dir = Path(save_root_dir)
        self.save_root_dir = save_root_dir
        self.clean_work_dir = clean_work_dir
        self.log_to_mlflow = log_to_mlflow
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.check_if_exists = check_if_exists
        self.parser = parser
        self.raise_on_fit_error = raise_on_fit_error
        self.error_score = error_score
        self.timeout_fit = timeout_fit
        self.timeout_combination = timeout_combination
        self.verbose = verbose
        self.client = None
        self.logger_filename = None

    @property
    def n_threads_per_task(self):
        if self._n_threads_per_task is None:
            return self.n_jobs
        return self._n_threads_per_task

    @property
    def n_gpus_per_task(self):
        if self._n_gpus_per_task is None:
            return self.n_gpus_per_worker
        return self._n_gpus_per_task

    @property
    @abstractmethod
    def models_dict(self):
        return {}

    def _validate_dict_of_models_params(self, dict_to_validate, models_nickname):
        """Validate the dictionary of models parameters or fit parameters, broadcasting it to the number of models
        if a single dictionary is passed instead of a dictionary of dictionaries."""
        dict_to_validate = dict_to_validate or {model_nickname: {} for model_nickname in models_nickname}
        models_missing = [model_nickname for model_nickname in models_nickname
                          if model_nickname not in dict_to_validate]
        if len(models_missing) == len(models_nickname):
            dict_to_validate = {model_nickname: dict_to_validate.copy() for model_nickname in models_nickname}
        else:
            for model_missing in models_missing:
                dict_to_validate[model_missing] = {}
        return dict_to_validate

    @abstractmethod
    def _add_arguments_to_parser(self):
        """Add the arguments to the parser."""
        self.parser.add_argument('--experiment_name', type=str, default=self.experiment_name)
        self.parser.add_argument('--create_validation_set', action='store_true')
        self.parser.add_argument('--models_nickname', type=str, choices=self.models_dict.keys(),
                                 nargs='*', default=self.models_nickname)
        self.parser.add_argument('--seeds_models', nargs='*', type=int, default=self.seeds_models)
        self.parser.add_argument('--n_jobs', type=int, default=self.n_jobs,
                                 help='Number of threads/cores to be used by the model if it supports it. Usually it is'
                                      'threads, but can depend on model.'
                                      'Obs.: In the CEREMADE cluster the minimum number of cores that can be requested'
                                      'are 2, so it is a good idea to set at least n_jobs to 2 if we want to use all '
                                      'the resources available.')
        self.parser.add_argument('--models_params', type=json.loads, default=self.models_params,
                                 help='Dictionary with the parameters of the models. The keys must be the nickname '
                                      'of the model and the values must be a dictionary with the parameters of '
                                      'the model. In case only one dictionary is passed, it will be used for '
                                      'all models. The dictionary is passed as a string in the json format, depending '
                                      'on your shell interpreter you may '
                                      'have to escape the quotes and not use spaces if you are using the command line.'
                                      'Examples of usage from cli: '
                                      '--models_params {\\"[MODEL_NICKNAME]\\":{\\"[NAME_PARAM_1]\\":[VALUE_PARAM_1],'
                                      '\\"[NAME_PARAM_2]\\":[VALUE_PARAM_2]},'
                                      '\\"[MODEL_NICKNAME_2]\\":{\\"[NAME_PARAM_1]\\":[VALUE_PARAM_1]}}'
                                      '--models_params {\\"[NAME_PARAM_1]\\":[VALUE_PARAM_1]}')
        self.parser.add_argument('--fits_params', type=json.loads, default=self.fits_params,
                                 help='Dictionary with the parameters of the fits. The keys must be the nickname '
                                      'of the model and the values must be a dictionary with the parameters of '
                                      'the fit. In case only one dictionary is passed, it will be used for '
                                      'all models. The dictionary is passed as a string in the json format, you may'
                                      'have to escape the quotes and not use spaces if you are using the command line. '
                                      'Please refer to the --models_params argument for examples of usage.')
        self.parser.add_argument('--error_score', type=str, default=self.error_score)
        self.parser.add_argument('--timeout_fit', type=int, default=self.timeout_fit)
        self.parser.add_argument('--timeout_combination', type=int, default=self.timeout_combination)
        self.parser.add_argument('--verbose', type=int, default=self.verbose)

        self.parser.add_argument('--log_dir', type=Path, default=self.log_dir)
        self.parser.add_argument('--log_file_name', type=str, default=self.log_file_name)
        self.parser.add_argument('--work_root_dir', type=Path, default=self.work_root_dir)
        self.parser.add_argument('--save_root_dir', type=Path, default=self.save_root_dir)
        self.parser.add_argument('--do_not_clean_work_dir', action='store_true')
        self.parser.add_argument('--do_not_log_to_mlflow', action='store_true')
        self.parser.add_argument('--mlflow_tracking_uri', type=str, default=self.mlflow_tracking_uri)
        self.parser.add_argument('--do_not_check_if_exists', action='store_true')
        self.parser.add_argument('--do_not_retry_on_oom', action='store_true')
        self.parser.add_argument('--raise_on_fit_error', action='store_true')

        self.parser.add_argument('--dask_cluster_type', type=str, default=self.dask_cluster_type)
        self.parser.add_argument('--n_workers', type=int, default=self.n_workers,
                                 help='Maximum number of workers to be used.')
        self.parser.add_argument('--n_cores_per_worker', type=int, default=self.n_cores_per_worker)
        self.parser.add_argument('--n_processes_per_worker', type=int, default=self.n_processes_per_worker)
        self.parser.add_argument('--n_threads_per_worker', type=int, default=self.n_threads_per_worker)
        self.parser.add_argument('--n_processes_per_task', type=int, default=self.n_processes_per_task)
        self.parser.add_argument('--n_cores_per_task', type=int, default=self.n_cores_per_task)
        self.parser.add_argument('--n_threads_per_task', type=int, default=self.n_threads_per_task)
        self.parser.add_argument('--dask_memory', type=str, default=self.dask_memory)
        self.parser.add_argument('--dask_job_extra_directives', type=str, default=self.dask_job_extra_directives)
        self.parser.add_argument('--dask_address', type=str, default=self.dask_address)
        self.parser.add_argument('--n_gpus_per_worker', type=float, default=self.n_gpus_per_worker)
        self.parser.add_argument('--n_gpus_per_task', type=float, default=self.n_gpus_per_task)

    @abstractmethod
    def _unpack_parser(self):
        """Unpack the parser."""
        args = self.parser.parse_args()
        self.experiment_name = args.experiment_name
        self.create_validation_set = args.create_validation_set
        self.models_nickname = args.models_nickname
        self.n_jobs = args.n_jobs
        models_params = args.models_params
        self.models_params = self._validate_dict_of_models_params(models_params, self.models_nickname)
        fits_params = args.fits_params
        self.fits_params = self._validate_dict_of_models_params(fits_params, self.models_nickname)
        error_score = args.error_score
        if error_score == 'nan':
            error_score = np.nan
        self.error_score = error_score
        self.timeout_fit = args.timeout_fit
        self.timeout_combination = args.timeout_combination
        self.verbose = args.verbose

        self.log_dir = args.log_dir
        self.log_file_name = args.log_file_name
        self.work_root_dir = args.work_root_dir
        self.save_root_dir = args.save_root_dir
        self.clean_work_dir = not args.do_not_clean_work_dir
        self.log_to_mlflow = not args.do_not_log_to_mlflow
        self.mlflow_tracking_uri = args.mlflow_tracking_uri
        self.check_if_exists = not args.do_not_check_if_exists
        self.raise_on_fit_error = args.raise_on_fit_error

        self.dask_cluster_type = args.dask_cluster_type
        self.n_workers = args.n_workers
        self.n_cores_per_worker = args.n_cores_per_worker
        self.n_processes_per_worker = args.n_processes_per_worker
        self.n_threads_per_worker = args.n_threads_per_worker
        self.n_processes_per_task = args.n_processes_per_task
        self.n_cores_per_task = args.n_cores_per_task
        self._n_threads_per_task = args.n_threads_per_task
        self.dask_memory = args.dask_memory
        dask_job_extra_directives = args.dask_job_extra_directives
        # parse dask_job_extra_directives
        if isinstance(dask_job_extra_directives, str):
            # the following was generated by chatgpt, it seems to work
            dask_job_extra_directives = shlex.split(dask_job_extra_directives)
            dask_job_extra_directives = [
                f"{dask_job_extra_directives[i]} {dask_job_extra_directives[i + 1]}"
                if i + 1 < len(dask_job_extra_directives) and not dask_job_extra_directives[i + 1].startswith('-')
                else dask_job_extra_directives[i]
                for i in range(len(dask_job_extra_directives)) if dask_job_extra_directives[i].startswith('-')
            ]
        else:
            dask_job_extra_directives = []
        self.dask_job_extra_directives = dask_job_extra_directives
        self.dask_address = args.dask_address
        self.n_gpus_per_worker = args.n_gpus_per_worker
        self._n_gpus_per_task = args.n_gpus_per_task
        return args

    def _treat_parser(self):
        """Treat the parser."""
        if self.parser is not None:
            self._add_arguments_to_parser()
            self._unpack_parser()

    def _setup_logger(self, log_dir=None, filemode='w'):
        """Setup the logger."""
        if log_dir is None:
            log_dir = self.log_dir
        os.makedirs(log_dir, exist_ok=True)
        if self.log_file_name is None:
            name = self.experiment_name
        else:
            name = self.log_file_name
        if (log_dir / f'{name}.log').exists():
            file_names = sorted(log_dir.glob(f'{name}_????.log'))
            if file_names:
                file_name = file_names[-1].name
                id_file = int(file_name.split('_')[-1].split('.')[0])
                name = f'{name}_{id_file + 1:04d}'
            else:
                name = name + '_0001'
        log_file_name = f'{name}.log'
        logging.basicConfig(filename=log_dir / log_file_name,
                            format='%(asctime)s - %(levelname)s\n%(message)s\n',
                            level=logging.INFO, filemode=filemode)

    def _setup_dask(self, n_workers, cluster_type='local', address=None):
        """Set up the dask cluster."""
        if address is not None:
            client = Client(address)
        else:
            # allow multiprocessing with joblib inside dask workers
            dask.config.set({'distributed.worker.daemon': False})
            if cluster_type == 'local':
                if n_workers * self.n_threads_per_worker > cpu_count():
                    warnings.warn(f"n_workers * n_threads_per_worker is greater than the number of "
                                  f"cores (checked with cpu_count()) "
                                  f"available ({cpu_count()}). This may lead to performance issues.")
                resources_per_worker = {'cores': self.n_cores_per_worker, 'gpus': self.n_gpus_per_worker,
                                        'processes': self.n_processes_per_worker, 'threads': self.n_threads_per_worker,
                                        '_whole_worker': 1}

                # set threads used by numpy / scipy (OpenMP, MKL, OpenBLAS)
                os.environ['OMP_NUM_THREADS'] = str(self.n_threads_per_worker)
                os.environ['MKL_NUM_THREADS'] = str(self.n_threads_per_worker)
                os.environ['OPENBLAS_NUM_THREADS'] = str(self.n_threads_per_worker)
                pre_env = {
                    'distributed.nanny.pre-spawn-environ.OMP_NUM_THREADS': self.n_threads_per_worker,
                    'distributed.nanny.pre-spawn-environ.MKL_NUM_THREADS': self.n_threads_per_worker,
                    'distributed.nanny.pre-spawn-environ.OPENBLAS_NUM_THREADS': self.n_threads_per_worker,
                }
                dask.config.set(pre_env)

                cluster = LocalCluster(n_workers=0, memory_limit=self.dask_memory, processes=True,
                                       threads_per_worker=self.n_threads_per_worker, resources=resources_per_worker,
                                       local_directory=self.work_root_dir)
                cluster.scale(n_workers)
            elif cluster_type == 'slurm':
                resources_per_work = {'cores': self.n_cores_per_worker, 'gpus': self.n_gpus_per_worker,
                                      'processes': self.n_processes_per_worker, 'threads': self.n_threads_per_worker,
                                      '_whole_worker': 1}

                # set threads used by numpy / scipy (OpenMP, MKL, OpenBLAS)
                os.environ['OMP_NUM_THREADS'] = str(self.n_threads_per_worker)
                os.environ['MKL_NUM_THREADS'] = str(self.n_threads_per_worker)
                os.environ['OPENBLAS_NUM_THREADS'] = str(self.n_threads_per_worker)
                pre_env = {
                    'distributed.nanny.pre-spawn-environ.OMP_NUM_THREADS': self.n_threads_per_worker,
                    'distributed.nanny.pre-spawn-environ.MKL_NUM_THREADS': self.n_threads_per_worker,
                    'distributed.nanny.pre-spawn-environ.OPENBLAS_NUM_THREADS': self.n_threads_per_worker,
                }
                dask.config.set(pre_env)

                job_extra_directives = dask.config.get(
                    "jobqueue.slurm.job-extra-directives", []
                )
                job_script_prologue = dask.config.get(
                    "jobqueue.slurm.job-script-prologue", []
                )
                worker_extra_args = dask.config.get(
                    "jobqueue.slurm.worker-extra-args", []
                )
                job_extra_directives = job_extra_directives + self.dask_job_extra_directives
                job_script_prologue = job_script_prologue + [f'export OMP_NUM_THREADS={self.n_threads_per_worker}'
                                                             f'export MKL_NUM_THREADS={self.n_threads_per_worker}'
                                                             f'export OPENBLAS_NUM_THREADS={self.n_threads_per_worker}'
                                                             f'eval "$(conda shell.bash hook)"',
                                                             f'conda activate $CONDA_DEFAULT_ENV']
                resources_per_work_string = ' '.join([f'{key}={value}' for key, value in resources_per_work.items()])
                worker_extra_args = worker_extra_args + [f'--resources "{resources_per_work_string}"']
                walltime = '364-23:59:59'
                job_name = f'dask-worker-{self.experiment_name}'
                # some slurm clusters are configured to use cores=cores and others cores=threads, which makes it really
                # difficult to know how to spawn a new job and what do we want, we will have to manually adjust
                # resource_per_worker and resource_per_task to correctly use this implementation in each slurm cluster
                cluster = SLURMCluster(cores=self.n_cores_per_worker, memory=self.dask_memory,
                                       processes=self.n_processes_per_worker,
                                       job_extra_directives=job_extra_directives,
                                       job_script_prologue=job_script_prologue, walltime=walltime,
                                       job_name=job_name, worker_extra_args=worker_extra_args)
                log_and_print_msg(f"Cluster script generated:\n{cluster.job_script()}", verbose=self.verbose,
                                  verbose_level=1)
                cluster.scale(n_workers)
            else:
                raise ValueError("cluster_type must be either 'local' or 'slurm'.")
            log_and_print_msg("Cluster dashboard address", dashboard_address=cluster.dashboard_link,
                              verbose=self.verbose, verbose_level=1)
            client = cluster.get_client()
        logging_plugin = LoggingSetterPlugin(logging_config={'level': logging.INFO})
        client.register_plugin(logging_plugin)
        mlflow_plugin = MLFlowCleanupPlugin()
        client.register_plugin(mlflow_plugin)
        client.forward_logging()
        return client

    def _on_train_start(self, combination: dict, unique_params: Optional[dict] = None,
                        extra_params: Optional[dict] = None, **kwargs):
        if self.n_gpus_per_task > 0:
            if torch_available:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                if self.n_gpus_per_task < 1:
                    # this will supposedly allow to use just a fraction of the gpu memory
                    # open question: what if we want to use more than one gpu?
                    torch.cuda.set_per_process_memory_fraction(self.n_gpus_per_task)
        return {}

    def _before_load_data(self, combination: dict, unique_params: Optional[dict] = None,
                          extra_params: Optional[dict] = None, **kwargs):
        return {}

    @abstractmethod
    def _load_data(self, combination: dict, unique_params: Optional[dict] = None,
                   extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _after_load_data(self, combination: dict, unique_params: Optional[dict] = None,
                         extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _before_load_model(self, combination: dict, unique_params: Optional[dict] = None,
                           extra_params: Optional[dict] = None, **kwargs):
        return {}

    @abstractmethod
    def _load_model(self, combination: dict, unique_params: Optional[dict] = None,
                    extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _after_load_model(self, combination: dict, unique_params: Optional[dict] = None,
                          extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _before_get_metrics(self, combination: dict, unique_params: Optional[dict] = None,
                            extra_params: Optional[dict] = None, **kwargs):
        return {}

    @abstractmethod
    def _get_metrics(self, combination: dict, unique_params: Optional[dict] = None,
                     extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _after_get_metrics(self, combination: dict, unique_params: Optional[dict] = None,
                           extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _before_fit_model(self, combination: dict, unique_params: Optional[dict] = None,
                          extra_params: Optional[dict] = None, **kwargs):
        return {}

    @abstractmethod
    def _fit_model(self, combination: dict, unique_params: Optional[dict] = None,
                   extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _after_fit_model(self, combination: dict, unique_params: Optional[dict] = None,
                         extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _before_evaluate_model(self, combination: dict, unique_params: Optional[dict] = None,
                               extra_params: Optional[dict] = None, **kwargs):
        return {}

    @abstractmethod
    def _evaluate_model(self, combination: dict, unique_params: Optional[dict] = None,
                        extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _after_evaluate_model(self, combination: dict, unique_params: Optional[dict] = None,
                              extra_params: Optional[dict] = None, **kwargs):
        return {}

    def _on_exception(self, combination: dict, unique_params: Optional[dict] = None,
                      extra_params: Optional[dict] = None, **kwargs):
        return self._on_exception_or_train_end(combination, unique_params, extra_params=extra_params, **kwargs)

    def _on_train_end(self, combination: dict, unique_params: Optional[dict] = None,
                      extra_params: Optional[dict] = None, **kwargs):
        return self._on_exception_or_train_end(combination, unique_params, extra_params=extra_params, **kwargs)

    def _on_exception_or_train_end(self, combination: dict, unique_params: Optional[dict] = None,
                                   extra_params: Optional[dict] = None, **kwargs):
        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        self._log_run_results(combination=combination, unique_params=unique_params, extra_params=extra_params,
                              mlflow_run_id=mlflow_run_id, **kwargs)

        # save and/or clean work_dir
        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        work_dir = self.get_local_work_dir(combination, mlflow_run_id, unique_params)
        if self.save_root_dir:
            # copy work_dir to save_dir
            if mlflow_run_id is not None:
                mlflow.log_artifacts(work_dir, artifact_path='work_dir', run_id=mlflow_run_id)
            else:
                save_dir = self.save_root_dir / work_dir.name
                copytree(work_dir, save_dir, dirs_exist_ok=True)
        if self.clean_work_dir:
            if work_dir.exists():
                rmtree(work_dir)
        return {}

    def _log_base_experiment_run_results(self, combination: dict, unique_params: Optional[dict] = None,
                                         extra_params: Optional[dict] = None, mlflow_run_id=None, **kwargs):
        if mlflow_run_id is None:
            return
        log_params = {}
        log_metrics = {}
        log_tags = {}
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)

        # log elapsed times
        for key, value in kwargs.items():
            if key == 'total_elapsed_time':
                log_metrics['total_elapsed_time'] = value
            elif isinstance(value, dict):
                elapsed_time = value.get('elapsed_time', None)
                if elapsed_time is not None:
                    log_metrics[key + '_elapsed_time'] = elapsed_time

        # log memory usage in MB (in linux getrusage seems to returns in KB)
        log_metrics['max_memory_used'] = getrusage(RUSAGE_SELF).ru_maxrss / 1000

        if torch_available:
            if torch.cuda.is_available():
                log_metrics['max_cuda_memory_reserved'] = torch.cuda.max_memory_reserved() / (1024 ** 2)  # in MB
                log_metrics['max_cuda_memory_allocated'] = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB

        # log exceptions and set run status
        if 'exception' in kwargs:
            exception = kwargs['exception']
            log_tags.update({'raised_exception': True, 'EXCEPTION': str(exception)})
            mlflow_client.set_terminated(mlflow_run_id, status='FAILED')
        else:
            log_tags.update({'raised_exception': False})
            mlflow_client.set_terminated(mlflow_run_id, status='FINISHED')

        # log model parameters
        if 'model' in kwargs.get('load_model_return', {}):
            model = kwargs['load_model_return']['model']
            model_params = model.get_params()
            model_params = flatten_dict(model_params)
            # Sanitize callables
            for param in model_params.keys():
                if callable(model_params[param]):
                    try:
                        model_params[param] = model_params[param].__name__
                    except AttributeError:
                        model_params[param] = str(model_params[param])
            log_params.update(model_params)

        mlflow.log_params(log_params, run_id=mlflow_run_id)
        mlflow.log_metrics(log_metrics, run_id=mlflow_run_id)
        for tag, value in log_tags.items():
            mlflow_client.set_tag(mlflow_run_id, tag, value)

    def _log_run_results(self, combination: dict, unique_params: Optional[dict] = None,
                         extra_params: Optional[dict] = None, mlflow_run_id=None, **kwargs):
        self._log_base_experiment_run_results(combination=combination, unique_params=unique_params,
                                              extra_params=extra_params, mlflow_run_id=mlflow_run_id, **kwargs)

    def get_local_work_dir(self, combination: dict, mlflow_run_id=None, unique_params: Optional[dict] = None):
        try:
            # if running on a dask worker, we use the worker's local directory as work_root_dir
            worker = get_worker()
            work_dir = Path(worker.local_directory)
        except ValueError:
            # if running on the main process, we use the work_root_dir defined in the class
            work_dir = self.work_root_dir

        if mlflow_run_id is not None:
            unique_name = mlflow_run_id
        else:
            run_unique_params = combination.copy()
            if unique_params is not None:
                run_unique_params.update(unique_params)
            unique_name = '_'.join([f'{key}_{value}' for key, value in run_unique_params.items()])

        work_dir = work_dir / unique_name
        os.makedirs(work_dir, exist_ok=True)
        return work_dir

    def _add_elapsed_time(self, fn, fn_return_dict=True, *args, **kwargs):
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        if fn_return_dict:
            if result is not None:
                result['elapsed_time'] = elapsed_time
            else:
                result = {'elapsed_time': elapsed_time}
            return result
        else:
            return result, elapsed_time

    def _treat_train_model_exception(self, exception, combination: dict, unique_params: Optional[dict] = None,
                                     extra_params: Optional[dict] = None, results: Optional[dict] = None,
                                     start_time: Optional[float] = None, return_results: bool = False,
                                     **kwargs):
        total_elapsed_time = time.perf_counter() - start_time
        results['total_elapsed_time'] = total_elapsed_time
        if isinstance(exception, FunctionTimedOut):
            exception_to_log = 'FunctionTimedOut'  # otherwise it is difficult to read the log
        else:
            exception_to_log = exception
        results['on_exception_return'] = self._add_elapsed_time(self._on_exception, exception=exception_to_log,
                                                                combination=combination,
                                                                unique_params=unique_params,
                                                                extra_params=extra_params, **kwargs, **results)
        log_and_print_msg('Error while running', verbose=self.verbose, verbose_level=1,
                          exception=exception_to_log, total_elapsed_time=total_elapsed_time,
                          **combination, **unique_params)
        if self.raise_on_fit_error:
            raise exception
        if return_results:
            try:
                return results
            except UnboundLocalError:
                return {}
        else:
            return False

    def _train_model(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                     return_results: bool = False, **kwargs):
        try:
            results = {}
            timeout_fit = extra_params.get('timeout_fit', None)
            log_and_print_msg('Running...', verbose=self.verbose, verbose_level=1, **combination, **unique_params)
            start_time = time.perf_counter()

            results['on_train_start_return'] = self._add_elapsed_time(self._on_train_start, combination=combination,
                                                                      unique_params=unique_params,
                                                                      extra_params=extra_params, **kwargs, **results)

            # load data
            results['before_load_data_return'] = self._add_elapsed_time(self._before_load_data, combination=combination,
                                                                        unique_params=unique_params,
                                                                        extra_params=extra_params, **kwargs, **results)

            results['load_data_return'] = self._add_elapsed_time(self._load_data, combination=combination,
                                                                 unique_params=unique_params,
                                                                 extra_params=extra_params, **kwargs, **results)
            results['after_load_data_return'] = self._add_elapsed_time(self._after_load_data, combination=combination,
                                                                       unique_params=unique_params,
                                                                       extra_params=extra_params, **kwargs, **results)

            # load model
            results['before_load_model_return'] = self._add_elapsed_time(self._before_load_model,
                                                                         combination=combination,
                                                                         unique_params=unique_params,
                                                                         extra_params=extra_params, **kwargs, **results)
            results['load_model_return'] = self._add_elapsed_time(self._load_model, combination=combination,
                                                                  unique_params=unique_params,
                                                                  extra_params=extra_params, **kwargs, **results)
            results['after_load_model_return'] = self._add_elapsed_time(self._after_load_model, combination=combination,
                                                                        unique_params=unique_params,
                                                                        extra_params=extra_params, **kwargs, **results)

            # get metrics
            results['before_get_metrics_return'] = self._add_elapsed_time(self._before_get_metrics,
                                                                          combination=combination,
                                                                          unique_params=unique_params,
                                                                          extra_params=extra_params, **kwargs,
                                                                          **results)
            results['get_metrics_return'] = self._add_elapsed_time(self._get_metrics, combination=combination,
                                                                   unique_params=unique_params,
                                                                   extra_params=extra_params, **kwargs, **results)
            results['after_get_metrics_return'] = self._add_elapsed_time(self._after_get_metrics,
                                                                         combination=combination,
                                                                         unique_params=unique_params,
                                                                         extra_params=extra_params, **kwargs, **results)

            # fit model
            results['before_fit_model_return'] = self._add_elapsed_time(self._before_fit_model, combination=combination,
                                                                        unique_params=unique_params,
                                                                        extra_params=extra_params, **kwargs, **results)

            if timeout_fit is not None:
                kwargs_fit_model = dict(fn=self._fit_model, combination=combination, unique_params=unique_params,
                                        extra_params=extra_params)
                kwargs_fit_model.update(kwargs)
                kwargs_fit_model.update(results)
                results['fit_model_return'] = func_timeout(timeout_fit, self._add_elapsed_time,
                                                           kwargs=kwargs_fit_model)
            else:
                results['fit_model_return'] = self._add_elapsed_time(self._fit_model, combination=combination,
                                                                     unique_params=unique_params,
                                                                     extra_params=extra_params, **kwargs, **results)
            results['after_fit_model_return'] = self._add_elapsed_time(self._after_fit_model, combination=combination,
                                                                       unique_params=unique_params,
                                                                       extra_params=extra_params, **kwargs, **results)

            # evaluate model
            results['before_evaluate_model_return'] = self._add_elapsed_time(self._before_evaluate_model,
                                                                             combination=combination,
                                                                             unique_params=unique_params,
                                                                             extra_params=extra_params, **kwargs,
                                                                             **results)
            results['evaluate_model_return'] = self._add_elapsed_time(self._evaluate_model, combination=combination,
                                                                      unique_params=unique_params,
                                                                      extra_params=extra_params, **kwargs, **results)
            results['after_evaluate_model_return'] = self._add_elapsed_time(self._after_evaluate_model,
                                                                            combination=combination,
                                                                            unique_params=unique_params,
                                                                            extra_params=extra_params, **kwargs,
                                                                            **results)
        except FunctionTimedOut as exception:
            # we need to catch FunctionTimedOut separately because it is not a subclass of Exception
            return self._treat_train_model_exception(exception, combination=combination, unique_params=unique_params,
                                                     extra_params=extra_params, results=results, start_time=start_time,
                                                     return_results=return_results, **kwargs)
        except Exception as exception:
            return self._treat_train_model_exception(exception, combination=combination, unique_params=unique_params,
                                                     extra_params=extra_params, results=results, start_time=start_time,
                                                     return_results=return_results, **kwargs)
        else:
            total_elapsed_time = time.perf_counter() - start_time
            results['total_elapsed_time'] = total_elapsed_time
            results['on_train_end_return'] = self._add_elapsed_time(self._on_train_end, combination=combination,
                                                                    unique_params=unique_params,
                                                                    extra_params=extra_params, **kwargs, **results)
            log_and_print_msg('Finished!', verbose=self.verbose, verbose_level=1,
                              total_elapsed_time=total_elapsed_time, **combination, **unique_params)
            if return_results:
                return results
            else:
                return True

    def _log_base_experiment_start_params(self, mlflow_run_id, **run_unique_params):
        params_to_log = flatten_dict(run_unique_params).copy()
        params_to_log.update(dict(
            git_hash=get_git_revision_hash(),
            # dask parameters
            dask_cluster_type=self.dask_cluster_type,
            n_workers=self.n_workers,
            n_cores_per_worker=self.n_cores_per_worker,
            n_processes_per_worker=self.n_processes_per_worker,
            n_threads_per_worker=self.n_threads_per_worker,
            n_processes_per_task=self.n_processes_per_task,
            n_cores_per_task=self.n_cores_per_task,
            n_threads_per_task=self.n_threads_per_task,
            dask_memory=self.dask_memory,
            dask_job_extra_directives=self.dask_job_extra_directives,
            dask_address=self.dask_address,
            n_gpus_per_worker=self.n_gpus_per_worker,
            n_gpus_per_task=self.n_gpus_per_task,
        ))
        tags_to_log = dict(
            # slurm parameters
            SLURM_JOB_ID=os.getenv('SLURM_JOB_ID', None),
            SLURM_STEP_ID=os.getenv('SLURM_STEP_ID', None),
            SLURM_ARRAY_JOB_ID=os.getenv('SLURM_ARRAY_JOB_ID', None),
            SLURM_ARRAY_TASK_ID=os.getenv('SLURM_ARRAY_TASK_ID', None),
            SLURM_LOCALID=os.getenv('SLURM_LOCALID', None),
            SLURMD_NODENAME=os.getenv('SLURMD_NODENAME', None),
        )
        mlflow.log_params(params_to_log, run_id=mlflow_run_id)
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        for tag, value in tags_to_log.items():
            mlflow_client.set_tag(mlflow_run_id, tag, value)

    def _log_run_start_params(self, mlflow_run_id, **run_unique_params):
        """Log the parameters of the run to mlflow."""
        self._log_base_experiment_start_params(mlflow_run_id, **run_unique_params)

    def _run_mlflow_and_train_model(self, combination: dict, mlflow_run_id=None,
                                    unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                                    return_results=False, **kwargs):

        run_unique_params = combination.copy()
        if unique_params is not None:
            run_unique_params.update(unique_params)
        possible_existent_run = set_mlflow_tracking_uri_check_if_exists(self.experiment_name, self.mlflow_tracking_uri,
                                                                        self.check_if_exists, **run_unique_params)
        if possible_existent_run is not None:
            log_and_print_msg('Run already exists on MLflow. Skipping...', verbose=self.verbose, verbose_level=1)
            if return_results:
                return possible_existent_run.to_dict()
            else:
                return True

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            if self.save_root_dir:
                artifact_location = str(self.save_root_dir / self.experiment_name)
            else:
                artifact_location = None
            experiment_id = mlflow.create_experiment(self.experiment_name, artifact_location=artifact_location)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(self.experiment_name)
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        if mlflow_run_id is None:
            run = mlflow_client.create_run(experiment_id)
            mlflow_run_id = run.info.run_id

        mlflow_client.update_run(mlflow_run_id, status='RUNNING')
        self._log_run_start_params(mlflow_run_id, **run_unique_params)
        if extra_params is not None:
            extra_params['mlflow_run_id'] = mlflow_run_id
        else:
            extra_params = {'mlflow_run_id': mlflow_run_id}

        return self._train_model(combination=combination, unique_params=unique_params, return_results=return_results,
                                 extra_params=extra_params, **kwargs)

    def _run_combination(self, *combination, combination_names: Optional[list[str]] = None,
                         unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                         return_results=False, **kwargs):
        combination_dict = dict(zip(combination_names, combination))
        timeout_combination = extra_params.get('timeout_combination', None)

        # this is ugly, but will work for the moment, in summary we want to find mlflow_run_id in extra_params
        # when inside _train_model
        mlflow_run_id = combination_dict.pop('mlflow_run_id', None)
        if mlflow_run_id is None:
            # we try to get from extra_params and in the last case from kwargs
            mlflow_run_id = extra_params.pop('mlflow_run_id', None)
            if mlflow_run_id is None:
                mlflow_run_id = kwargs.pop('mlflow_run_id', None)

        kwargs_fn = dict(combination=combination_dict, unique_params=unique_params,
                         extra_params=extra_params, return_results=return_results)
        kwargs_fn.update(kwargs)
        if self.log_to_mlflow:
            kwargs_fn['mlflow_run_id'] = mlflow_run_id
            fn = self._run_mlflow_and_train_model
        else:
            kwargs_fn['extra_params']['mlflow_run_id'] = mlflow_run_id
            fn = self._train_model
        if timeout_combination:
            try:
                return func_timeout(timeout_combination, fn, kwargs=kwargs_fn)
            except FunctionTimedOut as exception:
                if self.raise_on_fit_error:
                    raise exception
                if return_results:
                    return {}
                else:
                    return False
        else:
            return fn(**kwargs_fn)

    @abstractmethod
    def _get_combinations(self):
        """Get the combinations of the experiment."""
        combinations = []
        combination_names = []
        unique_params = dict()
        extra_params = dict()
        return combinations, combination_names, unique_params, extra_params

    def _create_mlflow_run(self, *combination, combination_names: Optional[list[str]] = None,
                           unique_params: Optional[dict] = None, extra_params: Optional[dict] = None):
        """Create a mlflow run."""
        combination_dict = dict(zip(combination_names, combination))
        run_unique_params = combination_dict.copy()
        if unique_params is not None:
            run_unique_params.update(unique_params)
        possible_existent_run = set_mlflow_tracking_uri_check_if_exists(self.experiment_name, self.mlflow_tracking_uri,
                                                                        self.check_if_exists, **run_unique_params)
        if possible_existent_run is not None:
            return possible_existent_run.run_id
        else:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                if self.save_root_dir:
                    artifact_location = str(self.save_root_dir / self.experiment_name)
                else:
                    artifact_location = None
                experiment_id = mlflow.create_experiment(self.experiment_name, artifact_location=artifact_location)
            else:
                experiment_id = experiment.experiment_id
            mlflow.set_experiment(self.experiment_name)
            mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            run = mlflow_client.create_run(experiment_id)
            mlflow_run_id = run.info.run_id
            mlflow_client.update_run(mlflow_run_id, status='SCHEDULED')
            return mlflow_run_id

    def _run_experiment(self, client=None):
        """Run the experiment."""

        combinations, combination_names, unique_params, extra_params = self._get_combinations()
        log_and_print_msg('Starting experiment...', verbose=self.verbose, verbose_level=1,
                          combination_names=combination_names, combinations=combinations,
                          unique_params=unique_params, extra_params=extra_params)
        n_args = len(combinations[0])

        total_combinations = len(combinations)
        n_combinations_successfully_completed = 0
        n_combinations_failed = 0
        n_combinations_none = 0
        if client is not None:
            first_args = list(combinations[0])
            list_of_args = [[combination[i] for combination in combinations[1:]] for i in range(n_args)]
            # we will first create the mlflow runs to avoid threading problems
            if self.log_to_mlflow:
                resources_per_task = {'_whole_worker': 1}  # ensure 1 worker creates 1 run
                first_future = client.submit(self._create_mlflow_run, *first_args, resources=resources_per_task,
                                             pure=False, combination_names=combination_names,
                                             unique_params=unique_params, extra_params=extra_params)
                futures = [first_future]
                if total_combinations > 1:
                    # we will wait a bit in case we need to create the experiment, directories, etc
                    time.sleep(5)
                    other_futures = client.map(self._create_mlflow_run, *list_of_args, pure=False,
                                               batch_size=self.n_workers, resources=resources_per_task,
                                               combination_names=combination_names,
                                               unique_params=unique_params, extra_params=extra_params)
                    futures.extend(other_futures)
                mlflow_run_ids = client.gather(futures)
                for future in futures:
                    future.release()  # release the memory of the future
                combinations = [list(combination) + [run_id]
                                for combination, run_id in zip(combinations, mlflow_run_ids)]
                combination_names.append('mlflow_run_id')
            if hasattr(self, 'n_trials'):
                # the resources are actually used when training the models, here we will launch the hpo framework
                # so we ensure that each worker launches only one hpo run
                resources_per_task = {'threads': 0, 'cores': 0, 'gpus': 0, 'processes': 0, '_whole_worker': 1}
            else:
                resources_per_task = {'threads': self.n_threads_per_task, 'cores': self.n_cores_per_task,
                                      'gpus': self.n_gpus_per_worker, 'processes': self.n_processes_per_task}
            log_and_print_msg(f'{total_combinations} models are being trained and evaluated in parallel, '
                              f'check the logs for real time information. We will display information about the '
                              f'completion of the tasks right after sending all the tasks to the cluster. '
                              f'Note that this can take a while if a lot of tasks are being submitted. '
                              f'You can check the dask dashboard to get more information about the progress and '
                              f'the workers.', verbose=self.verbose, verbose_level=1)

            workers = {str(value['name']): value['resources'] for worker_address, value
                       in client.scheduler_info()['workers'].items()}
            free_workers = list(workers.keys())
            futures = []
            submitted_combinations = 0
            finished_combinations = 0
            with tqdm(total=len(combinations), desc='Combinations completed') as progress_bar:
                while finished_combinations < total_combinations:
                    # submit tasks to free workers
                    while free_workers and submitted_combinations < total_combinations:
                        worker_name = free_workers.pop()
                        worker = workers[worker_name]
                        combination = list(combinations[submitted_combinations])
                        key = '_'.join(str(arg) for arg in combination)
                        future = client.submit(self._run_combination, *combination, pure=False, key=key,
                                               resources=resources_per_task, workers=[worker_name],
                                               allow_other_workers=True, combination_names=combination_names,
                                               unique_params=unique_params, extra_params=extra_params)
                        future.worker = worker_name
                        futures.append(future)
                        worker_can_still_work = True
                        for resource in resources_per_task:
                            worker[resource] -= resources_per_task[resource]
                            if worker[resource] < resources_per_task[resource]:
                                worker_can_still_work = False
                        if worker_can_still_work:
                            free_workers.append(worker_name)
                        submitted_combinations += 1

                    # wait for at least one task to finish
                    completed_future = next(as_completed(futures))
                    combination_success = completed_future.result()
                    if combination_success is True:
                        n_combinations_successfully_completed += 1
                    elif combination_success is False:
                        n_combinations_failed += 1
                    else:
                        n_combinations_none += 1
                    finished_combinations += 1
                    progress_bar.update(1)
                    log_and_print_msg(str(progress_bar), verbose=self.verbose, verbose_level=1,
                                      succesfully_completed=n_combinations_successfully_completed,
                                      failed=n_combinations_failed, none=n_combinations_none)
                    completed_worker_name = completed_future.worker
                    worker = workers[completed_worker_name]
                    worker_can_work = True
                    for resource in resources_per_task:
                        worker[resource] += resources_per_task[resource]
                        if worker[resource] < resources_per_task[resource]:
                            worker_can_work = False
                    if worker_can_work:
                        free_workers.append(completed_worker_name)
                    futures.remove(completed_future)
                    completed_future.release()  # release the memory of the future

            client.close()
        else:
            progress_bar = tqdm(combinations, desc='Combinations completed')
            for combination in progress_bar:
                run_id = self._create_mlflow_run(*combination, combination_names=combination_names,
                                                 unique_params=unique_params, extra_params=extra_params)
                combination_with_run_id = list(combination) + [run_id]
                combination_names.append('mlflow_run_id')
                combination_success = self._run_combination(*combination_with_run_id,
                                                            combination_names=combination_names,
                                                            unique_params=unique_params, extra_params=extra_params)
                if combination_success is True:
                    n_combinations_successfully_completed += 1
                elif combination_success is False:
                    n_combinations_failed += 1
                else:
                    n_combinations_none += 1
                log_and_print_msg(str(progress_bar), verbose=self.verbose, verbose_level=1,
                                  succesfully_completed=n_combinations_successfully_completed,
                                  failed=n_combinations_failed, none=n_combinations_none)

        return total_combinations, n_combinations_successfully_completed, n_combinations_failed, n_combinations_none

    def run(self):
        """Run the entire pipeline."""
        self._treat_parser()
        os.makedirs(self.work_root_dir, exist_ok=True)
        if self.save_root_dir:
            os.makedirs(self.save_root_dir, exist_ok=True)
        self._setup_logger()
        start_time = time.perf_counter()
        if self.dask_cluster_type is not None:
            client = self._setup_dask(self.n_workers, self.dask_cluster_type, self.dask_address)
        else:
            client = None
        total_combinations, n_combinations_successfully_completed, n_combinations_failed, n_combinations_none = (
            self._run_experiment(client=client))
        total_time = time.perf_counter() - start_time
        log_and_print_msg('Experiment finished!', verbose=self.verbose, verbose_level=1, total_elapsed_time=total_time,
                          total_combinations=total_combinations,
                          sucessfully_completed=n_combinations_successfully_completed,
                          failed=n_combinations_failed, none=n_combinations_none)
        logging.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = BaseExperiment(parser=parser)
    experiment.run()
