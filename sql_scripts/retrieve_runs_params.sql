CREATE EXTENSION IF NOT EXISTS tablefunc;
SELECT
    *
FROM
    crosstab(
        $$
        SELECT
        	runs.run_uuid,
            params."key",
            params.value
        FROM
            runs
        LEFT JOIN experiments exps ON
            runs.experiment_id = exps.experiment_id
        LEFT JOIN params ON
            params.run_uuid = runs.run_uuid
        WHERE
            params."key" IN ('model_nickname', 'dataset_name', 'seed_dataset', 'fold', 'random_seed')
        ORDER BY
            1
        $$,
        $$
        SELECT unnest(ARRAY['model_nickname', 'dataset_name', 'seed_dataset', 'fold', 'random_seed'])
        $$
    ) AS ct(
        run_uuid TEXT,
        model_nickname TEXT,
        dataset_name TEXT,
        seed_dataset TEXT,
        fold TEXT,
        random_seed TEXT
    );
