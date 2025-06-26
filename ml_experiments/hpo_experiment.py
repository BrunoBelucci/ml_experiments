from abc import ABC, abstractmethod
from typing import Optional
from optuna.study import Study
from optuna_integration import DaskStorage
from distributed import get_client
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from ml_experiments.base_experiment import BaseExperiment
from ml_experiments.tuners import OptunaTuner, DaskOptunaTuner
from ml_experiments.utils import flatten_dict, profile_time, profile_memory
from functools import partial


class HPOExperiment(BaseExperiment, ABC):

    def __init__(
        self,
        *args,
        hpo_framework: str = "optuna",
        # general
        n_trials: int = 30,
        timeout_hpo: int = 0,
        timeout_trial: int = 0,
        max_concurrent_trials: int = 1,
        hpo_seed: int = 0,
        # optuna
        sampler: str = "tpe",
        pruner: str = "none",
        direction: str = "minimize",
        hpo_metric: str = 'validation_score',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hpo_framework = hpo_framework
        # general
        self.n_trials = n_trials
        self.timeout_hpo = timeout_hpo
        self.timeout_trial = timeout_trial
        self.max_concurrent_trials = max_concurrent_trials
        self.hpo_seed = hpo_seed
        # optuna
        self.sampler = sampler
        self.pruner = pruner
        self.direction = direction
        self.hpo_metric = hpo_metric
        self.log_dir_dask = None

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        if self.parser is None:
            raise ValueError('Parser is not initialized, please call the constructor of the class first')
        self.parser.add_argument('--hpo_framework', type=str, default=self.hpo_framework)
        # general
        self.parser.add_argument('--n_trials', type=int, default=self.n_trials)
        self.parser.add_argument('--timeout_hpo', type=int, default=self.timeout_hpo)
        self.parser.add_argument('--timeout_trial', type=int, default=self.timeout_trial)
        self.parser.add_argument('--max_concurrent_trials', type=int, default=self.max_concurrent_trials)
        self.parser.add_argument('--hpo_seed', type=int, default=self.hpo_seed)
        # optuna
        self.parser.add_argument('--sampler', type=str, default=self.sampler)
        self.parser.add_argument('--pruner', type=str, default=self.pruner)
        self.parser.add_argument('--direction', type=str, default=self.direction)
        self.parser.add_argument('--hpo_metric', type=str, default=self.hpo_metric)

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.hpo_framework = args.hpo_framework
        # general
        self.n_trials = args.n_trials
        self.timeout_hpo = args.timeout_hpo
        self.timeout_trial = args.timeout_trial
        self.max_concurrent_trials = args.max_concurrent_trials
        self.hpo_seed = args.hpo_seed
        # optuna
        self.sampler = args.sampler
        self.pruner = args.pruner
        self.direction = args.direction
        self.hpo_metric = args.hpo_metric
        return args

    def _get_unique_params(self):
        unique_params = super()._get_unique_params()
        unique_params.update(
            {
                "hpo_framework": self.hpo_framework,
                "n_trials": self.n_trials,
                "timeout_hpo": self.timeout_hpo,
                "timeout_trial": self.timeout_trial,
                "max_concurrent_trials": self.max_concurrent_trials,
                "hpo_seed": self.hpo_seed,
                "sampler": self.sampler,
                "pruner": self.pruner,
                "direction": self.direction,
                "hpo_metric": self.hpo_metric,
            }
        )
        return unique_params

    def _load_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        results = {}
        hpo_framework = unique_params["hpo_framework"]
        n_trials = unique_params["n_trials"]
        timeout_hpo = unique_params["timeout_hpo"]
        timeout_trial = unique_params["timeout_trial"]
        max_concurrent_trials = unique_params["max_concurrent_trials"]
        hpo_seed = unique_params["hpo_seed"]
        sampler = unique_params["sampler"]
        pruner = unique_params["pruner"]

        if hpo_framework == 'optuna':
            tuner_kwargs = dict(
                sampler=sampler,
                pruner=pruner,
                n_trials=n_trials,
                timeout_total=timeout_hpo,
                timeout_trial=timeout_trial,
                seed=hpo_seed,
            )

            if self.dask_cluster_type is not None:
                n_threads_per_task = unique_params["n_threads_per_task"]
                n_cores_per_task = unique_params["n_cores_per_task"]
                n_gpus_per_worker = unique_params["n_gpus_per_worker"]
                n_processes_per_task = unique_params["n_processes_per_task"]
                storage = DaskStorage(client=get_client())
                tuner_kwargs["dask_client"] = "worker_client"
                tuner_kwargs["storage"] = storage
                tuner_kwargs["max_concurrent_trials"] = max_concurrent_trials
                tuner_kwargs["resources_per_task"] = {
                    "threads": n_threads_per_task,
                    "cores": n_cores_per_task,
                    "gpus": n_gpus_per_worker,
                    "processes": n_processes_per_task,
                }
                tuner_cls = DaskOptunaTuner
            else:
                tuner_cls = OptunaTuner

            tuner = tuner_cls(**tuner_kwargs)  # type: ignore
            results['tuner'] = tuner

        return results

    @abstractmethod
    def training_fn(
        self,
        trial_dict: dict,
        combination: dict,
        unique_params: dict,
        extra_params: dict,
        mlflow_run_id: Optional[str] = None,
        **kwargs,
    ) -> dict:
        raise NotImplementedError(
            "The training_fn method must be implemented in the subclass. "
            "It should handle the training of the model for a given trial."
        )

    def get_trial_fn(
        self, 
        study: Study,
        search_space: dict, 
        combination: dict,
        unique_params: dict,
        extra_params: dict,
        mlflow_run_id: Optional[str] = None,
        child_runs_ids: Optional[list] = None,
        **kwargs,
    ) -> dict:
        flatten_search_space = flatten_dict(search_space)
        trial = study.ask(flatten_search_space)
        trial_number = trial.number
        trial_key = "_".join([str(value) for value in combination.values()])
        trial_key = trial_key + f"-{trial_number}"  # unique key (trial number)
        child_run_id = child_runs_ids[trial_number] if child_runs_ids else None
        trial.set_user_attr('child_run_id', child_run_id)
        return dict(trial=trial, trial_key=trial_key, child_run_id=child_run_id)

    @abstractmethod
    def get_search_space(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ) -> dict:
        raise NotImplementedError(
            "The get_search_space method must be implemented in the subclass. "
            "It should return the search space for the hyperparameter optimization."
        )

    @abstractmethod
    def get_default_values(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ) -> list:
        raise NotImplementedError(
            "The get_default_values method must be implemented in the subclass. "
            "It should return the default values for the hyperparameters."
        )

    @profile_time(enable_based_on_attribute="profile_time")
    @profile_memory(enable_based_on_attribute="profile_memory")
    def _fit_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        tuner: OptunaTuner = kwargs.get('load_model_return', {}).get('tuner', None)
        n_trials = unique_params['n_trials']

        # objective and search space (distribution)
        if mlflow_run_id is not None:
            parent_run_id = mlflow_run_id
            parent_run = mlflow.get_run(parent_run_id)
            child_runs = parent_run.data.tags
            child_runs = {key: value for key, value in child_runs.items() if key.startswith('child_run_id_')}
            child_runs_ids = list(child_runs.values())
            child_runs_numbers = [int(key.split('_')[-1]) for key in child_runs.keys()]
            # sort child runs by trial number
            child_runs_ids = [id for _, id in sorted(zip(child_runs_numbers, child_runs_ids))]
        else:
            parent_run_id = None
            child_runs_ids = [None for _ in range(self.n_trials)]

        search_space = self.get_search_space(combination, unique_params, extra_params, mlflow_run_id, **kwargs)
        default_values = self.get_default_values(combination, unique_params, extra_params, mlflow_run_id, **kwargs)
        get_trial_fn = partial(
            self.get_trial_fn,
            search_space=search_space,
            combination=combination,
            unique_params=unique_params,
            extra_params=extra_params,
            mlflow_run_id=mlflow_run_id,
            child_runs_ids=child_runs_ids,
            **kwargs,
        )

        study = tuner.tune(
            training_fn=self.training_fn,
            search_space=search_space,
            direction=self.direction,
            metric=self.hpo_metric,
            enqueue_configurations=default_values,
            get_trial_fn=get_trial_fn,
            combination=combination,
            unique_params=unique_params,
            extra_params=extra_params,
            mlflow_run_id=mlflow_run_id,
            leave_pbar=False,
            **kwargs
        )

        grid_search_stopped = tuner.grid_search_stopped
        elapsed_time_timed_out = tuner.timed_out
        n_trials_effective = tuner.n_trials_effective

        if grid_search_stopped:
            if parent_run_id:
                mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                mlflow_client.set_tag(parent_run_id, 'grid_search_stopped', 'True')
                mlflow_client.set_tag(parent_run_id, 'n_trials_effective', n_trials_effective)
                for child_run_id in child_runs_ids:
                    if child_run_id is not None:  # this should always happen if we have a parent_run_id
                        run_status = mlflow_client.get_run(child_run_id).info.status
                        if run_status == 'SCHEDULED':
                            mlflow_client.set_tag(child_run_id, 'raised_exception', True)
                            mlflow_client.set_tag(child_run_id, 'EXCEPTION', 'Grid search stopped.')
                            mlflow_client.set_terminated(child_run_id, status='FAILED')
        elif elapsed_time_timed_out:
            if parent_run_id:
                mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                mlflow_client.set_tag(parent_run_id, 'elapsed_time_timed_out', 'True')
                mlflow_client.set_tag(parent_run_id, 'n_trials_effective', n_trials_effective)
                for child_run_id in child_runs_ids:
                    if child_run_id is not None: # this should always happen if we have a parent_run_id
                        run_status = mlflow_client.get_run(child_run_id).info.status
                        if run_status == 'SCHEDULED':
                            mlflow_client.set_tag(child_run_id, 'raised_exception', True)
                            mlflow_client.set_tag(child_run_id, 'EXCEPTION', 'Elapsed time timed out.')
                            mlflow_client.set_terminated(child_run_id, status='FAILED')
        return dict(study=study)

    def _get_metrics(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        return {}

    def _evaluate_model(
        self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
    ):
        study = kwargs['fit_model_return']['study']

        best_trial = study.best_trial
        best_trial_result = best_trial.user_attrs.get('result', dict())
        best_metric_results = {f'best/{metric}': value for metric, value in best_trial_result.items()}
        best_value = best_trial.value
        best_metric_results['best/value'] = best_value
        # if best_metric_results is empty it means that every trial failed, we will raise an exception
        if not best_metric_results:
            raise ValueError('Every trial failed, no best model was found')

        if mlflow_run_id is not None:
            params_to_log = {f'best/{param}': value for param, value in best_trial.params.items()}
            best_child_run_id = best_trial.user_attrs.get('child_run_id', None)
            params_to_log['best/child_run_id'] = best_child_run_id
            mlflow.log_params(params_to_log, run_id=mlflow_run_id)
        return best_metric_results

    def _create_mlflow_run(self, *combination, combination_names: list[str], unique_params: dict, extra_params: dict):
        parent_run_id = super()._create_mlflow_run(
            *combination, combination_names=combination_names, unique_params=unique_params, extra_params=extra_params
        )
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        experiment = mlflow_client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f'Experiment {self.experiment_name} not found in mlflow')
        experiment_id = experiment.experiment_id
        # we will initialize the nested runs from the trials
        for trial in range(self.n_trials):
            run = mlflow_client.create_run(experiment_id, tags={MLFLOW_PARENT_RUN_ID: parent_run_id})
            run_id = run.info.run_id
            mlflow_client.set_tag(parent_run_id, f'child_run_id_{trial}', run_id)
            mlflow_client.update_run(run_id, status='SCHEDULED')
        return parent_run_id
