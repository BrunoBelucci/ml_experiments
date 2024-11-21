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
