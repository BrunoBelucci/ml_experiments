import subprocess

import mlflow


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
