import argparse
import time
from abc import ABC, abstractmethod
from math import floor
from typing import Optional
import optuna
from optuna_integration import DaskStorage
from distributed import get_client, worker_client
import mlflow
from warnings import warn
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from ml_experiments.base_experiment import BaseExperiment
from sklearn.utils import check_random_state
from tqdm.auto import tqdm
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut


def discretize_search_space(search_space):
    discretized_search_space = {}
    for param_name, optuna_distribution in search_space.items():
        if isinstance(optuna_distribution, optuna.distributions.BaseDistribution):
            if isinstance(optuna_distribution, optuna.distributions.CategoricalDistribution):
                discretized_search_space[param_name] = optuna_distribution.choices
            else:
                if isinstance(optuna_distribution, optuna.distributions.IntDistribution):
                    dtype = np.int64
                elif isinstance(optuna_distribution, optuna.distributions.FloatDistribution):
                    dtype = np.float64
                low = optuna_distribution.low
                high = optuna_distribution.high
                step = optuna_distribution.step
                log = optuna_distribution.log
                if not log:
                    n_samples = floor((high - low) / step) + 1
                    # use linspace instead of arange to avoid floating point errors as mentioned in
                    # https://numpy.org/doc/stable/reference/generated/numpy.arange.html
                    discretized_search_space[param_name] = np.linspace(low, high, n_samples, dtype=dtype,
                                                                       endpoint=True).tolist()
                else:
                    n_samples = floor(np.log(high / low) / np.log(step)) + 1
                    discretized_search_space[param_name] = np.logspace(np.log10(low), np.log10(high), n_samples,
                                                                       dtype=dtype, endpoint=True).tolist()
        else:
            pass
    return discretized_search_space


class HPOExperiment(BaseExperiment, ABC):
    def __init__(self, *args,
                 hpo_framework='optuna',
                 # general
                 n_trials=30, timeout_hpo=10 * 60 * 60, timeout_trial=2 * 60 * 60, max_concurrent_trials=1,
                 # optuna
                 sampler='tpe', pruner='hyperband', direction='minimize', hpo_metric=None,
                 **kwargs):
        """HPO experiment.

        This class allows to perform experiments with HPO for tabular data.It allows to perform experiments with
        different models, datasets and
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
        datasets_names_or_ids :
            The names or ids of the datasets to be used in the experiment. Defaults to None.
        seeds_datasets :
            The seeds to be used in the datasets. Defaults to None.
        resample_strategy :
            The resampling strategy to be used. Defaults to 'k-fold_cv'.
        k_folds :
            The number of folds to be used in the k-fold cross-validation. Defaults to 10.
        folds :
            The folds to be used in the resampling. Defaults to None.
        pct_test :
            The percentage of the test set. Defaults to 0.2.
        validation_resample_strategy :
            The resampling strategy to be used to create the validation set. Defaults to 'next_fold'.
        pct_validation :
            The percentage of the validation set. Defaults to 0.1.
        tasks_ids :
            The ids of the tasks to be used in the experiment. Defaults to None.
        task_repeats :
            The repeats to be used in the tasks. Defaults to None.
        task_folds :
            The folds to be used in the tasks. Defaults to None.
        task_samples :
            The samples to be used in the tasks. Defaults to None.
        experiment_name :
            The name of the experiment. Defaults to 'base_experiment'.
        create_validation_set :
            If True, create a validation set. Defaults to False.
        models_dict :
            The dictionary with the models to be used in the experiment, it must be a dictionary with the keys being
            the nickname of the model and the values being another dictionary with the class of the model and the
            parameters of the model.
        hpo_framework :
            The hyperparameter optimization framework to be used. It must be 'optuna' for the moment.
        n_trials :
            The number of trials to be run.
        timeout_hpo :
            The timeout of the experiment in seconds.
        timeout_trial :
            The timeout of each trial in seconds.
        max_concurrent_trials :
            The maximum number of concurrent trials that can be run.
        sampler :
            The sampler to be used in the hyperparameter optimization. It can be 'tpe' or 'random'.
        pruner :
            The pruner to be used in the hyperparameter optimization. It can be 'hyperband', 'sha' or None.
        """
        super().__init__(*args, **kwargs)
        self.hpo_framework = hpo_framework
        # general
        self.n_trials = n_trials
        self.timeout_hpo = timeout_hpo
        self.timeout_trial = timeout_trial
        self.max_concurrent_trials = max_concurrent_trials
        # optuna
        self.sampler = sampler
        self.pruner = pruner
        self.direction = direction
        self.hpo_metric = hpo_metric
        self.log_dir_dask = None

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--hpo_framework', type=str, default=self.hpo_framework)
        # general
        self.parser.add_argument('--n_trials', type=int, default=self.n_trials)
        self.parser.add_argument('--timeout_hpo', type=int, default=self.timeout_hpo)
        self.parser.add_argument('--timeout_trial', type=int, default=self.timeout_trial)
        self.parser.add_argument('--max_concurrent_trials', type=int, default=self.max_concurrent_trials)
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
        # optuna
        self.sampler = args.sampler
        self.pruner = args.pruner
        self.direction = args.direction
        self.hpo_metric = args.hpo_metric

    @abstractmethod
    def get_hyperband_max_resources(self, combination: dict, unique_params: Optional[dict] = None,
                                    extra_params: Optional[dict] = None, **kwargs):
        raise NotImplementedError('This method must be implemented in the subclass')

    def _load_model(self, combination: dict, unique_params: Optional[dict] = None,
                    extra_params: Optional[dict] = None, **kwargs):
        results = {}
        model_nickname = combination['model_nickname']
        model_cls = self.models_dict[model_nickname][0]
        seed_model = combination['seed_model']

        if self.hpo_framework == 'optuna':
            # sampler
            if self.sampler == 'tpe':
                sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed_model)
            elif self.sampler == 'random':
                sampler = optuna.samplers.RandomSampler(seed=seed_model)
            elif self.sampler == 'grid':
                search_space, default_values = model_cls.create_search_space()
                search_space = discretize_search_space(search_space)
                sampler = optuna.samplers.GridSampler(search_space=search_space, seed=seed_model)
            else:
                raise NotImplementedError(f'Sampler {self.sampler} not implemented for optuna')
            results['sampler'] = sampler

            # pruner
            if self.pruner == 'hyperband':
                max_resources = self.hyperband_max_resources
                n_brackets = 5
                min_resources = 1
                # the following should give us the desired number of brackets
                reduction_factor = floor((max_resources / min_resources) ** (1 / (n_brackets - 1)))
                pruner = optuna.pruners.HyperbandPruner(min_resource=min_resources, max_resource=max_resources,
                                                        reduction_factor=reduction_factor)
            elif self.pruner == 'sha':
                pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=10)
            elif self.pruner is None or (self.pruner).lower() == 'none':
                pruner = None
            else:
                raise NotImplementedError(f'Pruner {self.pruner} not implemented for optuna')
            results['pruner'] = pruner

            # storage
            if self.dask_cluster_type is not None:
                client = get_client()
                storage = DaskStorage(client=client)
            else:
                storage = None
            results['storage'] = storage

            study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, direction=self.direction)
            results['study'] = study
        else:
            raise NotImplementedError(f'HPO framework {self.hpo_framework} not implemented')

        return results

    def _training_fn(self, single_experiment: BaseExperiment, trial_combination: dict, optuna_trial: optuna.Trial,
                     unique_params: Optional[dict] = None, extra_params: Optional[dict] = None, **kwargs):
        combination_values = list(trial_combination.values())
        combination_names = list(trial_combination.keys())
        extra_params = extra_params.copy()
        parent_run_id = extra_params.pop('mlflow_run_id', None)
        timeout_trial = extra_params.pop('timeout_trial', self.timeout_trial)
        unique_params = unique_params.copy()
        # we actually need to consider parent_run_id as a unique parameter, because it will be used
        # exclusively for the parent_run, we cannot use a run from another parent_run for hpo
        unique_params['parent_run_id'] = parent_run_id
        if timeout_trial == 0:
            results = single_experiment._run_combination(*combination_values, combination_names=combination_names,
                                                         unique_params=unique_params,
                                                         extra_params=extra_params,
                                                         return_results=True)
        else:
            fn = single_experiment._run_combination
            kwargs_fn = dict(combination_names=combination_names, unique_params=unique_params,
                             extra_params=extra_params, return_results=True)
            try:
                results = func_timeout(timeout_trial, fn, args=combination_values, kwargs=kwargs_fn)
                if not isinstance(results, dict):
                    results = {'evaluate_model_return': {}}
                else:
                    if 'evaluate_model_return' not in results:
                        results['evaluate_model_return'] = {}
            except FunctionTimedOut:
                results = {'evaluate_model_return': {}}

        # we do not need to keep all the results (data, model...), only the evaluation results
        child_run_id = trial_combination.get('mlflow_run_id', None)
        keep_results = {'evaluate_model_return': results['evaluate_model_return'], 'child_run_id': child_run_id}
        return keep_results

    def _get_optuna_params(self, search_space, study, model_params, fit_params, combination, child_run_id,
                           random_state):
        optuna_distributions_search_space = {}
        conditional_distributions_search_space = {}
        for name, value in search_space.items():
            if isinstance(value, optuna.distributions.BaseDistribution):
                optuna_distributions_search_space[name] = value
            else:
                conditional_distributions_search_space[name] = value
        trial = study.ask(optuna_distributions_search_space)
        conditional_params = {name: fn(trial) for name, fn
                              in conditional_distributions_search_space.items()}
        trial_model_params = trial.params
        trial_model_params.update(model_params.copy())
        trial_seed_model = random_state.randint(0, 10000)
        trial_combination = combination.copy()
        trial_combination.pop('model_params')
        trial_combination.pop('seed_model')
        trial_combination.pop('fit_params')
        trial_key = '_'.join([str(value) for value in trial_combination.values()])  # shared prefix
        trial_key = trial_key + f'-{child_run_id}'  # unique key (child_run_id)
        trial_combination['model_params'] = trial_model_params
        trial_combination['seed_model'] = trial_seed_model
        trial_combination['fit_params'] = fit_params.copy()
        trial_combination['mlflow_run_id'] = child_run_id
        return trial, trial_combination, trial_key

    @abstractmethod
    def _load_single_experiment(self, combination: dict, unique_params: Optional[dict] = None,
                                extra_params: Optional[dict] = None, **kwargs):
        raise NotImplementedError('This method must be implemented in the subclass')

    @abstractmethod
    def _get_tell_metric_from_results(self, results):
        raise NotImplementedError('This method must be implemented in the subclass')

    def _get_tell_metric_from_results(self, results):
        evaluate_model_return = results.get('evaluate_model_return', {})
        hpo_metric = evaluate_model_return.get(self.hpo_metric, None)
        if hpo_metric is None:
            warn(f'hpo_metric {self.hpo_metric} not found in evaluate_model_return')
            if self.direction == 'maximize':
                return -float('inf')
            else:
                return float('inf')
        return evaluate_model_return[self.hpo_metric]

    def _fit_model(self, combination: dict, unique_params: Optional[dict] = None,
                   extra_params: Optional[dict] = None, **kwargs):
        model_nickname = combination['model_nickname']
        model_params = combination['model_params']
        fit_params = combination['fit_params']
        seed_model = combination['seed_model']
        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        model_cls = self.models_dict[model_nickname][0]
        timeout_hpo = unique_params.get('timeout_hpo', self.timeout_hpo)

        if self.hpo_framework == 'optuna':
            study = kwargs['load_model_return']['study']
            storage = kwargs['load_model_return']['storage']
            sampler = kwargs['load_model_return']['sampler']

            if isinstance(sampler, optuna.samplers.GridSampler):
                # we will ignore n_trials and run every trial defined in the search_space
                search_space = sampler._search_space
                n_samples_params = [len(value) for value in search_space.values()]
                n_trials = np.prod(n_samples_params)
            else:
                n_trials = self.n_trials

            # objective and search space (distribution)
            search_space, default_values = model_cls.create_search_space()
            random_state = check_random_state(seed_model)
            if mlflow_run_id is not None:
                parent_run_id = mlflow_run_id
                parent_run = mlflow.get_run(parent_run_id)
                child_runs = parent_run.data.tags
                child_runs_ids = [child_run_id for key, child_run_id in child_runs.items()
                                  if key.startswith('child_run_id_')]
                if len(child_runs_ids) < n_trials:  # runs were not created before, so we create them now
                    mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                    experiment_id = mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id
                    for trial in range(len(child_runs_ids) + 1, n_trials + 1):
                        run = mlflow_client.create_run(experiment_id, tags={MLFLOW_PARENT_RUN_ID: parent_run_id})
                        run_id = run.info.run_id
                        mlflow_client.set_tag(parent_run_id, f'child_run_id_{trial}', run_id)
                        mlflow_client.set_tag(run_id, 'parent_run_id', parent_run_id)
                        mlflow_client.update_run(run_id, status='SCHEDULED')
                        child_runs_ids.append(run_id)
            else:
                parent_run_id = None
                child_runs_ids = [None for _ in range(self.n_trials)]

            if not isinstance(sampler, optuna.samplers.GridSampler):
                study.enqueue_trial(default_values)

            # we will run several experiments
            single_experiment = self._load_single_experiment(combination, unique_params=unique_params,
                                                             extra_params=extra_params, **kwargs)
            grid_search_stopped = False
            elapsed_time_timed_out = False
            n_trial = 0
            first_trial = True
            start_time = time.perf_counter()
            pbar = tqdm(total=n_trials, desc='Trials')
            while n_trial < n_trials:
                if self.dask_cluster_type is not None:
                    with worker_client() as client:
                        futures = []
                        trial_numbers = []
                        for _ in range(self.max_concurrent_trials):
                            child_run_id = child_runs_ids[n_trial]
                            trial, trial_combination, trial_key = self._get_optuna_params(
                                search_space, study, model_params, fit_params, combination, child_run_id, random_state
                            )
                            trial_numbers.append(trial.number)
                            resources = {'cores': self.n_jobs, 'gpus': self.n_gpus / (self.n_cores / self.n_jobs)}
                            futures.append(
                                client.submit(self._training_fn, resources=resources, key=trial_key, pure=False,
                                              single_experiment=single_experiment, trial_combination=trial_combination,
                                              optuna_trial=trial, unique_params=unique_params,
                                              extra_params=extra_params, **kwargs)
                            )
                            n_trial += 1
                            if n_trial >= self.n_trials or first_trial:
                                # we have already enqueued all the trials, or it is the first trial,
                                # and we want to run it before the others
                                first_trial = False
                                break

                        results = client.gather(futures)
                        for future in futures:
                            future.release()

                    for trial_number, result in zip(trial_numbers, results):
                        study_id = storage.get_study_id_from_name(study.study_name)
                        trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, trial_number)
                        child_run_id = result['child_run_id']
                        storage.set_trial_user_attr(trial_id, 'child_run_id', child_run_id)
                        eval_result = result['evaluate_model_return']
                        for metric, value in eval_result.items():
                            storage.set_trial_user_attr(trial_id, metric, value)
                        try:
                            study.tell(trial_number, self._get_tell_metric_from_results(result))
                        except RuntimeError:  # handle stop of grid search
                            if not grid_search_stopped:  # we save the n_trial when the grid search stopped
                                n_trials_effective = n_trial
                            grid_search_stopped = True
                            pass
                        pbar.update(1)
                        if parent_run_id:
                            eval_result.pop('elapsed_time', None)
                            mlflow.log_metrics(eval_result, run_id=parent_run_id, step=trial_number)
                    elapsed_time = time.perf_counter() - start_time
                    if elapsed_time > timeout_hpo:
                        elapsed_time_timed_out = True
                        n_trials_effective = n_trial
                        break
                else:
                    child_run_id = child_runs_ids[n_trial]
                    trial, trial_combination, _ = self._get_optuna_params(
                        search_space, study, model_params, fit_params, combination, child_run_id, random_state
                    )
                    result = self._training_fn(single_experiment=single_experiment, trial_combination=trial_combination,
                                               optuna_trial=trial, unique_params=unique_params,
                                               extra_params=extra_params, **kwargs)
                    child_run_id = result['child_run_id']
                    trial.set_user_attr('child_run_id', child_run_id)
                    eval_result = result['evaluate_model_return']
                    for metric, value in eval_result.items():
                        trial.set_user_attr(metric, value)
                    try:
                        study.tell(trial, self._get_tell_metric_from_results(result))
                    except RuntimeError:  # handle stop of grid search
                        if not grid_search_stopped:  # we save the n_trial when the grid search stopped
                            n_trials_effective = n_trial
                        grid_search_stopped = True
                        pass
                    pbar.update(1)
                    if parent_run_id:
                        eval_result.pop('elapsed_time', None)
                        mlflow.log_metrics(eval_result, run_id=parent_run_id, step=trial.number)
                    n_trial += 1
                    elapsed_time = time.perf_counter() - start_time
                    if elapsed_time > timeout_hpo:
                        elapsed_time_timed_out = True
                        n_trials_effective = n_trial
                        break

            pbar.close()

            if grid_search_stopped:
                if parent_run_id:
                    mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
                    mlflow_client.set_tag(parent_run_id, 'grid_search_stopped', 'True')
                    mlflow_client.set_tag(parent_run_id, 'n_trials_effective', n_trials_effective)
                    for child_run_id in child_runs_ids:
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
                        run_status = mlflow_client.get_run(child_run_id).info.status
                        if run_status == 'SCHEDULED':
                            mlflow_client.set_tag(child_run_id, 'raised_exception', True)
                            mlflow_client.set_tag(child_run_id, 'EXCEPTION', 'Elapsed time timed out.')
                            mlflow_client.set_terminated(child_run_id, status='FAILED')

            return {}

        else:
            raise NotImplementedError(f'HPO framework {self.hpo_framework} not implemented')

    def _evaluate_model(self, combination: dict, unique_params: Optional[dict] = None,
                        extra_params: Optional[dict] = None, **kwargs):
        study = kwargs['load_model_return']['study']

        best_trial = study.best_trial
        best_metric_results = {f'best_{metric}': value for metric, value in best_trial.user_attrs.items()
                               if not metric.startswith('elapsed_') and not metric.startswith('child_run_id')}
        # if best_metric_results is empty it means that every trial failed, we will raise an exception
        if not best_metric_results:
            raise ValueError('Every trial failed, no best model was found')

        mlflow_run_id = extra_params.get('mlflow_run_id', None)
        if mlflow_run_id is not None:
            best_model_params = best_trial.params.copy()
            best_model_params['best_child_run_id'] = best_trial.user_attrs.get('child_run_id', None)
            mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            for tag, value in best_model_params.items():
                mlflow_client.set_tag(mlflow_run_id, tag, value)

        return best_metric_results

    def _get_combinations(self):
        combinations, combination_names, unique_params, extra_params = super()._get_combinations()
        unique_params.update(dict(hpo_framework=self.hpo_framework, n_trials=self.n_trials,
                                  timeout_hpo=self.timeout_hpo, timeout_trial=self.timeout_trial,
                                  max_concurrent_trials=self.max_concurrent_trials, sampler=self.sampler,
                                  pruner=self.pruner, create_validation_set=self.create_validation_set,
                                  direction=self.direction, hpo_metric=self.hpo_metric))
        if not self.create_validation_set:
            warn('HPOExperiment usually requires a validation set, are you sure you did not forgot set'
                 ' create_validation_set=True or pass --create_validation_set')
        return combinations, combination_names, unique_params, extra_params

    def _create_mlflow_run(self, *combination, combination_names: Optional[list[str]] = None,
                           unique_params: Optional[dict] = None, extra_params: Optional[dict] = None):
        parent_run_id = super()._create_mlflow_run(*combination, combination_names=combination_names,
                                                   unique_params=unique_params, extra_params=extra_params)
        mlflow_client = mlflow.client.MlflowClient(tracking_uri=self.mlflow_tracking_uri)
        experiment_id = mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id
        # we will initialize the nested runs from the trials
        for trial in range(self.n_trials):
            run = mlflow_client.create_run(experiment_id, tags={MLFLOW_PARENT_RUN_ID: parent_run_id})
            run_id = run.info.run_id
            mlflow_client.set_tag(parent_run_id, f'child_run_id_{trial}', run_id)
            mlflow_client.set_tag(run_id, 'parent_run_id', parent_run_id)
            mlflow_client.update_run(run_id, status='SCHEDULED')
        return parent_run_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOExperiment(parser=parser)
    experiment.run()
