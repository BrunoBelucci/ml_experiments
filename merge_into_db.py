from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy.exc import IntegrityError
import pandas as pd
from functools import partial


n_processes = 40
db2_engine = create_engine('postgresql://localhost:5432/default', pool_size=n_processes, max_overflow=0)
db1_urls = ['postgresql://beluccib@localhost:5430/default', 'postgresql://ude66tz@localhost:5431/default']
tables = ['experiments', 'runs', 'metrics', 'params', 'tags', 'latest_metrics']


def initializer():
    """ensure the parent proc's database connections are not touched
    in the new connection pool"""
    db2_engine.dispose(close=False)


# we are directly creating the function here with global variables db2_engine and tables because we are using
# multiprocessing and we cannot pickle the engine object
# This is more or less the recommended way to use multiprocessing with sqlalchemy
# consult: https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
experiments_id_map_db1_to_db2 = {}
def process_df(df, table, try_to_insert_new=False):
    if table.name == 'experiments':
        df_db1_experiments = df
        df_db2_experiments = pd.read_sql_table(table.name, db2_engine)
        # if name exists in db2, we will keep the experiments row from db2, and we will map the experiment_id
        # from db1 to db2
        experiments_db2 = set(df_db2_experiments['name'])
        experiments_db1 = set(df_db1_experiments['name'])
        common_experiments = experiments_db1.intersection(experiments_db2)
        for experiment in common_experiments:
            experiment_id_db1 = \
            df_db1_experiments.loc[df_db1_experiments['name'] == experiment, 'experiment_id'].values[0]
            experiment_id_db2 = \
            df_db2_experiments.loc[df_db2_experiments['name'] == experiment, 'experiment_id'].values[0]
            experiments_id_map_db1_to_db2[experiment_id_db1] = experiment_id_db2
        # remove common experiments from df
        df = df.loc[~df['name'].isin(common_experiments)]
        # if name does not exist in db2, we will keep the experiments row from db1, but we will assign a new
        # experiment_id which will be the max experiment_id in db2 + 1
        new_experiments = experiments_db1 - experiments_db2
        max_experiment_id_db2 = df_db2_experiments['experiment_id'].max()
        for experiment in new_experiments:
            experiment_id_db1 = \
            df_db1_experiments.loc[df_db1_experiments['name'] == experiment, 'experiment_id'].values[0]
            max_experiment_id_db2 += 1
            experiments_id_map_db1_to_db2[experiment_id_db1] = max_experiment_id_db2
            # assign new experiment_id
            df.loc[df['name'] == experiment, 'experiment_id'] = max_experiment_id_db2

    if table.name in ['runs', 'datasets', 'experiments_tags', 'trace_info']:
        df['experiment_id'] = df['experiment_id'].map(experiments_id_map_db1_to_db2)

    try:
        df.to_sql(table.name, db2_engine, if_exists="append", index=False, method="multi")
    except IntegrityError as error:
        if not try_to_insert_new:
            return db1_url, error
        else:
            # we will try to identify duplicated rows and insert only the new ones, for this we will read the
            # table from db2 and compare the run_uuids
            if 'run_uuid' not in df.columns:
                raise ValueError(f"Table {table.name} does not have a run_uuid column")
            run_uuids_db1 = set(df['run_uuid'])
            select_query = f"SELECT * FROM {table.name} WHERE run_uuid IN {tuple(run_uuids_db1)}"
            df_db2 = pd.read_sql_query(select_query, db2_engine)
            run_uuids_db2 = set(df_db2['run_uuid'])
            new_run_uuids = run_uuids_db1 - run_uuids_db2
            df_new = df.loc[df['run_uuid'].isin(new_run_uuids)]
            df_new.to_sql(table.name, db2_engine, if_exists="append", index=False, method="multi")
    return db1_url, None


def mlflow_db1_into_db2(db1_url, db2_engine=db2_engine, tables=tables, exclude_tables=None, try_to_insert_new=False,
                        chunksize=100000, n_processes=n_processes):
    # Possible problems to keep in mind:
    # What if db1 and db2 have the same run_uuid, I don't know if it is possible that mlflow generates the exact
    # run_uuid in two different databases, but it is possible if we are for example merging a database that has already
    # been merged. For now this will fail, and we will have to manually handle this case.
    if exclude_tables is None:
        exclude_tables = ['alembic_version']

    db1_engine = create_engine(db1_url, pool_size=n_processes, max_overflow=0)
    metadata = MetaData()
    metadata.reflect(bind=db2_engine)

    for table in metadata.sorted_tables:

        if tables and table.name not in tables:
            continue

        if exclude_tables and table.name in exclude_tables:
            continue

        print('Processing table:', table.name)

        try:
            df_iterator = pd.read_sql_table(table.name, db1_engine, chunksize=chunksize)
        except Exception as error:
            return db1_url, error

        with Pool(n_processes, initializer=initializer) as p:
            pbar = tqdm()
            process_df_partial = partial(process_df, table=table, try_to_insert_new=try_to_insert_new)
            for res in p.imap_unordered(process_df_partial, df_iterator):
                db1_url, error = res
                if error:
                    pbar.close()
                    return db1_url, error
                pbar.update(1)
            pbar.close()
    return db1_url, None


db1_urls_with_errors = {}
for db1_url in db1_urls:
    print('Processing db1:', db1_url)
    db1_url, error = mlflow_db1_into_db2(db1_url)
    if error:
        db1_urls_with_errors[db1_url] = error

if db1_urls_with_errors:
    print('Some db1 urls could not be merged into db2, check db1_urls_with_errors.csv for details')
    db1_urls_with_errors_file = Path() / 'db1_urls_with_errors.csv'
    df_error = pd.DataFrame(db1_urls_with_errors.items(), columns=['db1_url', 'error'])
    df_error.to_csv(db1_urls_with_errors_file, index=False)
