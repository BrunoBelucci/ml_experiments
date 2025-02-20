from sqlalchemy import create_engine, MetaData
import pandas as pd


def mlflow_db1_into_db2(db1_url, db2_url, exclude_tables=None):
    # Possible problems to keep in mind:
    # What if db1 and db2 have the same run_uuid, I don't know if it is possible that mlflow generates the exact
    # run_uuid in two different databases, but it is possible if we are for example merging a database that has already
    # been merged. For now this will fail, and we will have to manually handle this case.
    if exclude_tables is None:
        exclude_tables = ['alembic_version']
    # Create db_1 engine
    db1_engine = create_engine(db1_url)

    # Create the db_2 engine
    db2_engine = create_engine(db2_url)
    metadata = MetaData()
    metadata.reflect(bind=db2_engine)

    experiments_id_map_db1_to_db2 = {}

    for table in metadata.sorted_tables:
        if exclude_tables and table.name in exclude_tables:
            continue

        df = pd.read_sql_table(table.name, db1_engine)

        if table.name == 'experiments':
            df_db1_experiments = df
            df_db2_experiments = pd.read_sql_table(table.name, db2_engine)
            # if name exists in db2, we will keep the experiments row from db2, and we will map the experiment_id
            # from db1 to db2
            experiments_db2 = set(df_db2_experiments['name'])
            experiments_db1 = set(df_db1_experiments['name'])
            common_experiments = experiments_db1.intersection(experiments_db2)
            for experiment in common_experiments:
                experiment_id_db1 = df_db1_experiments.loc[df_db1_experiments['name'] == experiment, 'experiment_id'].values[0]
                experiment_id_db2 = df_db2_experiments.loc[df_db2_experiments['name'] == experiment, 'experiment_id'].values[0]
                experiments_id_map_db1_to_db2[experiment_id_db1] = experiment_id_db2
            # remove common experiments from df
            df = df.loc[~df['name'].isin(common_experiments)]
            # if name does not exist in db2, we will keep the experiments row from db1, but we will assign a new
            # experiment_id which will be the max experiment_id in db2 + 1
            new_experiments = experiments_db1 - experiments_db2
            max_experiment_id_db2 = df_db2_experiments['experiment_id'].max()
            for experiment in new_experiments:
                experiment_id_db1 = df_db1_experiments.loc[df_db1_experiments['name'] == experiment, 'experiment_id'].values[0]
                max_experiment_id_db2 += 1
                experiments_id_map_db1_to_db2[experiment_id_db1] = max_experiment_id_db2
                # assign new experiment_id
                df.loc[df['name'] == experiment, 'experiment_id'] = max_experiment_id_db2

        if table.name in ['runs', 'datasets', 'experiments_tags', 'trace_info']:
            df['experiment_id'] = df['experiment_id'].map(experiments_id_map_db1_to_db2)

        df.to_sql(table.name, db2_engine, if_exists="append", index=False, method="multi")
