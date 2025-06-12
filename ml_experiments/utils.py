import subprocess
import mlflow
from time import perf_counter
from memory_profiler import memory_usage
from functools import wraps


def flatten_dict(dct, parent_key='', sep='/'):
    """
    Flatten a dictionary.

    Parameters:
    dct (dict): Dictionary to be flattened.
    parent_key (str): Key of the parent dictionary.
    sep (str): Separator to be used between keys.

    Returns:
    dict: Flattened dictionary.
    """
    items = []
    for k, v in dct.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(dct, sep='/'):
    """
    Unflatten a dictionary.

    Parameters:
    dct (dict): Dictionary to be unflattened.
    sep (str): Separator used between keys.

    Returns:
    dict: Unflattened dictionary.
    """
    result = {}
    for k, v in dct.items():
        keys = k.split(sep)
        d = result
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = v
    return result


def update_recursively(dct_1, dct_2):
    """Update dictionary dct_1 with values from dct_2, recursively."""
    for key, value in dct_2.items():
        if isinstance(value, dict):
            dct_1[key] = update_recursively(dct_1.get(key, {}), value)
        else:
            dct_1[key] = value
    return dct_1


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        return 'Not a git repository'


def check_if_exists_mlflow(experiment_name, **kwargs):
    filter_string = " AND ".join([f'params."{k}" = "{v}"' for k, v in flatten_dict(kwargs).items()])
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_string)
    # remove ./mlruns if it is automatically created
    # if os.path.exists('./mlruns'):
    #     os.rmdir('./mlruns')
    if 'tags.raised_exception' in runs.columns:
        runs = runs.loc[(runs['status'] == 'FINISHED') & (runs['tags.raised_exception'] == 'False')]
        if not runs.empty:
            return runs.iloc[0]
        else:
            return None
    else:
        return None


def set_mlflow_tracking_uri_check_if_exists(experiment_name, mlflow_tracking_uri, check_if_exists, **kwargs):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if check_if_exists:
        run = check_if_exists_mlflow(experiment_name, **kwargs)
    else:
        run = None
    if run is not None:
        return run
    else:
        return None


def profile_time(return_in_dict=True, enable=True, enable_based_on_attribute=None):
    """
    Decorator factory to profile the execution time of a function.
    
    Parameters:
    func (callable): Function to be profiled.
    return_in_dict (bool): If True and if the function returns a dictionary, the execution time will be added to 
    the dictionary, otherwise it will be returned as a tuple.
    enable (bool): If False, disables profiling.
    enable_based_on_attribute (str): If provided, we assume that we are decorating a method of a class and we look for
      this attribute to determine whether to enable profiling.
    
    Returns:
    callable: Wrapped function that profiles execution time.
    """
    def decorator(func):
        if not enable:
            return func
        @wraps(func)
        def wrapper(*args, **kwargs):

            if enable_based_on_attribute is not None:
                maybe_self = args[0]
                try:
                    dynamic_enable = getattr(maybe_self, enable_based_on_attribute)
                except AttributeError:
                    raise AttributeError(f"Object {maybe_self} does not have attribute '{enable_based_on_attribute}'")
            else:
                dynamic_enable = True
            if not dynamic_enable:
                return func(*args, **kwargs)
            
            start_time = perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = perf_counter() - start_time
            if return_in_dict and isinstance(result, dict):
                result['elapsed_time'] = elapsed_time
                return result
            else:
                return result, elapsed_time
        return wrapper
    return decorator


def profile_memory(return_in_dict=True, enable=True, enable_based_on_attribute=None, **kwargs):
    """
    Decorator factory to profile the memory usage of a function.
    
    Parameters:
    func (callable): Function to be profiled.
    return_in_dict (bool): If True and if the function returns a dictionary, the memory usage will be added to 
    the dictionary, otherwise it will be returned as a tuple.
    enable (bool): If False, disables profiling.
    enable_based_on_attribute (str): If provided, we assume that we are decorating a method of a class and we look for
      this attribute to determine whether to enable profiling.
    **kwargs: Additional keyword arguments for memory_usage function.
    
    Returns:
    callable: Wrapped function that profiles memory usage.
    """
    def decorator(func):
        if not enable:
            return func
        max_usage = kwargs.pop('max_usage', True)
        retval = kwargs.pop('retval', True)
        def wrapper(*args, **kwargs_from_func):

            if enable_based_on_attribute is not None:
                maybe_self = args[0]
                try:
                    dynamic_enable = getattr(maybe_self, enable_based_on_attribute)
                except AttributeError:
                    raise AttributeError(f"Object {maybe_self} does not have attribute '{enable_based_on_attribute}'")
            else:
                dynamic_enable = True
            if not dynamic_enable:
                return func(*args, **kwargs_from_func)
            
            if retval:
                mem_usage, result = memory_usage((func, args, kwargs_from_func), max_usage=max_usage, retval=retval, **kwargs)
            else:
                mem_usage = memory_usage((func, args, kwargs_from_func), max_usage=max_usage, retval=retval, **kwargs)
                result = None
            if return_in_dict and isinstance(result, dict):
                if max_usage:
                    result['max_memory_used'] = mem_usage
                else:
                    result['memory_usage'] = mem_usage
                return result
            else:
                if retval:
                    return result, mem_usage
                else:
                    return mem_usage
        return wrapper
    return decorator
