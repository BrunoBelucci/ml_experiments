from abc import ABC, abstractmethod
from typing import Optional
import optuna
from optuna.samplers import BaseSampler
from optuna.pruners import BasePruner
from optuna.storages import BaseStorage, InMemoryStorage
from optuna_integration import DaskStorage
from time import perf_counter
from tqdm.auto import tqdm
from ml_experiments.utils import flatten_dict
from dask.distributed import Client


class OptunaTuner(ABC):
    def __init__(
            self,
            sampler: str | type[BaseSampler] = "tpe",
            sampler_kwargs: Optional[dict] = None,
            pruner: Optional[str | type[BasePruner]] = None,
            pruner_kwargs: Optional[dict] = None,
            storage: Optional[str | type[BaseStorage]] = None,
            storage_kwargs: Optional[dict] = None,
            n_trials: int = 100,
            timeout_total: Optional[int] = None,
            seed: Optional[int] = None,
    ):
        self.sampler = sampler
        self.sampler_kwargs = sampler_kwargs if sampler_kwargs is not None else {}
        self.pruner = pruner
        self.pruner_kwargs = pruner_kwargs if pruner_kwargs is not None else {}
        self.storage = storage
        self.storage_kwargs = storage_kwargs if storage_kwargs is not None else {}
        self.n_trials = n_trials
        self.timeout_total = timeout_total
        self.seed = seed
        self.n_trials_effective = None
        self.study = None

    def get_sampler(self):
        if isinstance(self.sampler, str):
            if self.sampler == 'tpe':
                sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=self.seed)
            elif self.sampler == 'random':
                sampler = optuna.samplers.RandomSampler(seed=self.seed)
            # elif self.sampler == 'grid':
            #     search_space, default_values = model_cls.create_search_space()
            #     search_space = discretize_search_space(search_space)
            #     sampler = optuna.samplers.GridSampler(search_space=search_space, seed=self.seed_model)
            else:
                raise NotImplementedError(f'Sampler {self.sampler} not implemented for optuna')
        elif isinstance(self.sampler, type) and issubclass(self.sampler, BaseSampler):
            sampler = self.sampler(**self.sampler_kwargs)
        else:
            raise ValueError(f'Invalid sampler type: {type(self.sampler)}')
        return sampler
    
    def get_pruner(self):
        if isinstance(self.pruner, str):
            # if self.pruner == 'hyperband':
            #     max_resources = self.hyperband_max_resources
            #     n_brackets = 5
            #     min_resources = 1
            #     # the following should give us the desired number of brackets
            #     reduction_factor = floor((max_resources / min_resources) ** (1 / (n_brackets - 1)))
            #     pruner = optuna.pruners.HyperbandPruner(min_resource=min_resources, max_resource=max_resources,
            #                                             reduction_factor=reduction_factor)
            if self.pruner == 'sha':
                pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=10)
            elif (self.pruner).lower() == 'none':
                pruner = None
            else:
                raise NotImplementedError(f'Pruner {self.pruner} not implemented for optuna')
        elif isinstance(self.pruner, type) and issubclass(self.pruner, BasePruner):
            pruner = self.pruner(**self.pruner_kwargs)
        elif self.pruner is None:
            pruner = None
        else:
            raise ValueError(f'Invalid pruner type: {type(self.pruner)}')
        return pruner
    
    def get_storage(self):
        if self.storage is None:
            storage = InMemoryStorage()
        elif isinstance(self.storage, type) and issubclass(self.storage, BaseStorage):
            storage = self.storage(**self.storage_kwargs)
        else:
            raise ValueError(f'Invalid storage type: {type(self.storage)}')
        return storage
    
    def get_trial(self, search_space):
        if self.study is None:
            raise RuntimeError("self.study is not initialized. Please create the study before calling get_trial.")
        flatten_search_space = flatten_dict(search_space)
        trial = self.study.ask(flatten_search_space)
        return trial

    def run_simple_sequential_trial(self, training_fn, search_space, **kwargs):
        results = []
        trials_numbers = []
        trial = self.get_trial(search_space=search_space)
        result = training_fn(trial, **kwargs)
        trial_number = trial.number
        trials_numbers.append(trial_number)
        results.append(result)
        return trials_numbers, results

    def run_trials(self, training_fn, search_space, **kwargs):
        return self.run_simple_sequential_trial(training_fn=training_fn, search_space=search_space, **kwargs)

    def tune(self, training_fn, search_space, direction="minimize", metric=None, **kwargs):
        sampler = self.get_sampler()
        pruner = self.get_pruner()
        storage = self.get_storage()
        self.study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, direction=direction)
        n_trial = 0
        grid_search_stopped = False
        timed_out = False
        start_time = perf_counter()
        pbar = tqdm(total=self.n_trials, desc='Trials')
        while n_trial < self.n_trials:
            trials_numbers, results = self.run_trials(training_fn=training_fn, search_space=search_space, **kwargs)
            n_trials_run = len(trials_numbers)
            n_trial += n_trials_run
            for trial_number, result in zip(trials_numbers, results):
                study_id = storage.get_study_id_from_name(self.study.study_name)
                trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, trial_number)
                storage.set_trial_user_attr(trial_id, 'result', result)
                if isinstance(result, dict):
                    if metric is not None:
                        tell_value = result[metric]
                    else:
                        raise ValueError("Metric must be specified if training function returns a dict")
                else:
                    tell_value = result
                try:
                    self.study.tell(trial_number, tell_value)
                except RuntimeError:  # handle stop of grid search
                    grid_search_stopped = True
                    break
            pbar.update(n_trials_run)
            elapsed_time = perf_counter() - start_time
            if self.timeout_total is not None:
                if elapsed_time > self.timeout_total:
                    print(f"Timeout reached: {self.timeout_total} seconds")
                    timed_out = True
                    self.n_trials_effective = n_trial
                    break
            if grid_search_stopped:
                print(f"Grid search stopped after {n_trial} trials")
                self.n_trials_effective = n_trial
                break
        if not grid_search_stopped and not timed_out:
            self.n_trials_effective = n_trial
        pbar.close()
        return self.study
    

class DaskOptunaTuner(OptunaTuner):
    def __init__(
            self,
            *args,
            max_concurrent_trials: int = 1,
            dask_client: Optional[Client] = None,
            resources_per_task: Optional[dict] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_concurrent_trials = max_concurrent_trials
        self.dask_client = dask_client
        self.resources_per_task = resources_per_task if resources_per_task is not None else {}

    def get_storage(self):
        if self.storage is None:
            storage = InMemoryStorage()
        elif isinstance(self.storage, str):
            if self.storage == 'dask':
                storage = DaskStorage(client=self.dask_client)
            else:
                raise ValueError(f'Invalid storage string: {self.storage}')
        elif isinstance(self.storage, type) and issubclass(self.storage, BaseStorage):
            storage = self.storage(**self.storage_kwargs)
        else:
            raise ValueError(f'Invalid storage type: {type(self.storage)}')
        return storage
    
    def run_trials(self, training_fn, search_space, trial_key_prefix=None, **kwargs): # type: ignore
        """
        Run trials in parallel using Dask.
        """
        futures = []
        trials_numbers = []
        if self.dask_client is None:
            raise RuntimeError("Dask client is not initialized. Please create the Dask client before calling run_trials.")
        for _ in range(self.max_concurrent_trials):
            trial = self.get_trial(search_space=search_space)
            if trial_key_prefix is not None:
                trial_key = f"{trial_key_prefix}-{trial.number}"
            else:
                trial_key = f"trial-{trial.number}"
            trials_numbers.append(trial.number)
            future = self.dask_client.submit(training_fn, resources=self.resources_per_task, key=trial_key,
                                             pure=False, trial=trial, **kwargs)
            futures.append(future)
        results = self.dask_client.gather(futures)
        for future in futures:
            future.release()
        return trials_numbers, results
    