from itertools import product
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


def get_df_runs_from_mlflow_sql(engine, runs_columns, experiments_columns=None, experiments_names=None,
                                other_table=None, other_table_keys=None):
    if 'run_uuid' not in runs_columns:
        runs_columns.append('run_uuid')
    if experiments_columns is None:
        experiments_columns = []
    if experiments_names is None:
        experiments_names = []
    if other_table_keys is None:
        other_table_keys = []
    query = "SELECT "
    run_columns_for_query = [f"runs.{col}" for col in runs_columns]
    query_run_columns = ", ".join(run_columns_for_query)
    query += query_run_columns
    if experiments_columns:
        experiment_columns_for_query = [f"experiments.{col}" for col in experiments_columns]
        query_experiment_columns = ", " + ", ".join(experiment_columns_for_query)
        query += query_experiment_columns
    if other_table:
        query_other_table_columns = f', {other_table}."key", {other_table}.value'
        query += query_other_table_columns
    query += " FROM runs"
    if experiments_columns or experiments_names:
        query += " LEFT JOIN experiments ON runs.experiment_id = experiments.experiment_id"
    if other_table:
        query += f" LEFT JOIN {other_table} ON runs.run_uuid = {other_table}.run_uuid"
    if experiments_names or other_table_keys:
        query += " WHERE"
        if experiments_names:
            experiment_names_for_query = [f"'{name}'" for name in experiments_names]
            experiment_names_for_query = ", ".join(experiment_names_for_query)
            query += f' experiments.name IN ({experiment_names_for_query})'
            if other_table_keys:
                query += " AND"
        if other_table_keys:
            other_table_keys_for_query = [f"'{key}'" for key in other_table_keys]
            other_table_keys_for_query = ", ".join(other_table_keys_for_query)
            query += f' {other_table}."key" IN ({other_table_keys_for_query})'
    df = pd.read_sql(query, engine)
    if other_table_keys:
        df = df.pivot(columns='key', index=runs_columns + experiments_columns, values='value').reset_index()
    df = df.set_index('run_uuid')
    return df


def get_missing_entries(df, columns_names, should_contain_values):
    df = df.copy()
    indexes = product(*should_contain_values)
    df_should_contain = pd.DataFrame(index=indexes)
    contain = [df[column_name] for column_name in columns_names]
    df['indexes'] = list(zip(*contain))
    df_missing = df.join(df_should_contain, 'indexes', how='right')
    df_missing = df_missing.loc[df_missing[columns_names[0]].isna()]
    df_missing = pd.DataFrame(df_missing.indexes.to_list(), columns=columns_names)
    return df_missing


def get_common_combinations(df, column, combination_columns):
    df = df.copy()
    column_values = df[column].unique()
    values_combinations = {}
    for column_value in column_values:
        df_column_value = df.loc[df[column] == column_value]
        combinations = df_column_value[combination_columns].itertuples(index=False, name=None)
        values_combinations[column_value] = set(combinations)
    common_combinations = set.intersection(*values_combinations.values())
    return common_combinations


def get_df_with_combinations(df, combination_columns, combinations):
    df = df.copy()
    df['combination'] = list(df[combination_columns].itertuples(index=False, name=None))
    df = df.loc[df['combination'].isin(combinations)]
    df = df.drop(columns='combination')
    return df


def get_dfs_means_stds_both(df, column_model_name, column_task_id, column_dataset_name, column_metric):
    df = df.copy()
    columns = [column_model_name, column_task_id, column_dataset_name, column_metric]
    means = (
        df[columns].
        groupby(columns[:-1]).
        mean().
        reset_index(level=column_model_name).
        pivot(columns=column_model_name).
        droplevel(level=0,axis=1)
    )
    stds = (
        df[columns].
        groupby(columns[:-1]).
        std().
        reset_index(level=column_model_name).
        pivot(columns=column_model_name).
        droplevel(level=0,axis=1)
    )
    dfs = [means, stds]
    both = pd.concat([dfs[0], dfs[1]], keys=['mean', 'std'], axis=1)
    both.columns = both.columns.swaplevel(0, 1)
    both = both.sort_index(axis=1)
    return dfs[0], dfs[1], both


def friedman_nemenyi_test(df, model_column, block_column, metric_column, ascending_rank=False, alpha=0.95):
    df = df.copy()
    groups = [df.loc[df[model_column] == model, metric_column] for model in pd.unique(df[model_column])]
    res_friedman = friedmanchisquare(*groups)
    if res_friedman.pvalue < 1 - alpha:
        res_nemenyi = sp.posthoc_nemenyi_friedman(df, y_col=metric_column, block_col=block_column,
                                                  group_col=model_column, melted=True)
        ranks = df.groupby([block_column])[metric_column].rank(ascending=ascending_rank)
        df['rank'] = ranks
        mean_rank = df.groupby(model_column)['rank'].mean()
        return res_friedman, res_nemenyi, mean_rank
    else:
        print(f'pvalue of {res_friedman.pvalue} is greater than alpha of {alpha}')
        return res_friedman, None, None
