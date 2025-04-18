import argparse
from pathlib import Path
from inspect import cleandoc
import os


def generate_postgres_db_script(
        conda_env, db_name, database_root_dir,
        file_dir=Path.cwd(),
        file_name='start_db',
        db_port=5001, mlflow_port=5002, generate_sbatch=True,
        # sbatch parameters
        n_cores=6, clust_name='clust9',
        output_job_file=None,
        error_job_file=None,
        wall_time='364-23:59:59',
):
    if isinstance(file_dir, str):
        file_dir = Path(file_dir)
    os.makedirs(file_dir, exist_ok=True)
    if isinstance(database_root_dir, str):
        database_root_dir = Path(database_root_dir)
    database_dir = database_root_dir / (db_name + '_db')
    job_name = f'{db_name}_db'
    if output_job_file is None:
        output_job_dir = database_root_dir / 'sbatch_outputs'
        os.makedirs(output_job_dir, exist_ok=True)
        output_job_file = str(output_job_dir / '%x.%J.out')
    if error_job_file is None:
        error_job_dir = database_root_dir / 'sbatch_errors'
        os.makedirs(error_job_dir, exist_ok=True)
        error_job_file = str(error_job_dir / '%x.%J.err')
    log_file = database_dir / (db_name + '.log')
    sh_content = cleandoc(f"""
    if [ ! -d {str(database_dir.absolute())} ]; then
        conda run -n {conda_env} initdb -D {str(database_dir.absolute())}
        echo "host	all	all	samenet	trust" >> {str(database_dir.absolute())}/pg_hba.conf
    fi
    conda run -n {conda_env} pg_ctl -D {str(database_dir.absolute())} -l {str(log_file.absolute())} -o "-h 0.0.0.0 -p {db_port}" start
    conda run -n {conda_env} createdb {db_name} -p {db_port}
    conda run -n {conda_env} mlflow server --backend-store-uri postgresql://localhost:{db_port}/{db_name} -h 0.0.0.0 -p {mlflow_port}
    """) + '\n'
    if generate_sbatch:
        sbatch_content = cleandoc(f"""
        #!/bin/sh
        #SBATCH -c {n_cores}
        #SBATCH -w {clust_name}
        #SBATCH --job-name={job_name}
        #SBATCH --output={output_job_file}
        #SBATCH --error={error_job_file}
        #SBATCH --time={wall_time}
        """) + '\n'
        file_content = sbatch_content + sh_content
        file_ext = '.sbatch'
    else:
        file_content = sh_content
        file_ext = '.sh'
    file_path = file_dir / (file_name + file_ext)
    with open(file_path, 'w') as file:
        file.write(file_content)
    return file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conda_env', type=str, required=True)
    parser.add_argument('--database_root_dir', type=str, required=True)
    parser.add_argument('--db_name', type=str, required=True)
    parser.add_argument('--file_dir', type=str, default=Path.cwd())
    parser.add_argument('--file_name', type=str, default='start_db')
    parser.add_argument('--db_port', type=int, default=5001)
    parser.add_argument('--mlflow_port', type=int, default=5002)
    parser.add_argument('--generate_sbatch', type=bool, default=True)
    parser.add_argument('--n_cores_per_worker', type=int, default=6)
    parser.add_argument('--clust_name', type=str, default='clust9')
    parser.add_argument('--output_job_file', type=str)
    parser.add_argument('--error_job_file', type=str)
    parser.add_argument('--wall_time', type=str, default='364-23:59:59')
    args = parser.parse_args()
    generate_postgres_db_script(**vars(args))
