SELECT
    *
FROM
    crosstab(
        $$
        SELECT
        	runs.run_uuid,
            lame."key",
            lame.value
        FROM
            runs
        LEFT JOIN latest_metrics lame ON
            lame.run_uuid = runs.run_uuid
        WHERE
            lame."key" IN ('test_rmse', 'test_r2_score')
        ORDER BY
            1
        $$,
        $$
        SELECT unnest(ARRAY['test_rmse', 'test_r2_score'])
        $$
    ) AS ct(
        run_uuid TEXT,
        test_rmse float,
        test_r2_score float
    );
