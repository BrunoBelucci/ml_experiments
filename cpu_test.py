from multiprocessing import cpu_count
from datetime import datetime
import os
from subprocess import run

print(f"""
Now: {datetime.now()}
Number of CPUs (cpu_count): {cpu_count()}
Number of CPUs (affinity): {len(os.sched_getaffinity(0))}
CPUs affinity: {os.sched_getaffinity(0)}
{run(['bash', '-c', 'grep Cpus_allowed_list /proc/self/status'], capture_output=True).stdout.decode().strip()}
Slurm CPU Bind Type: {os.environ.get('SLURM_CPU_BIND_TYPE')}
Slurm CPU Bind List: {os.environ.get('SLURM_CPU_BIND_LIST')}
Slurm CPUs on Node: {os.environ.get('SLURM_CPUS_ON_NODE')}
Slurm CPUs per task: {os.environ.get('SLURM_CPUS_PER_TASK')}
Slurm TRES per task: {os.environ.get('SLURM_TRES_PER_TASK')}
Slurm Node Name: {os.environ.get('SLURMD_NODENAME')}
Slurm Job Start Time: {os.environ.get('SLURM_JOB_START_TIME')}
Slurm Job ID: {os.environ.get('SLURM_JOB_ID')}
Slurm Step ID: {os.environ.get('SLURM_STEP_ID')}
Slurm N tasks: {os.environ.get('SLURM_NTASKS')}
Slurm task PID: {os.environ.get('SLURM_TASK_PID')}
Slurm local ID: {os.environ.get('SLURM_LOCALID')}
""")
