import os
import argparse
from shutil import copyfile
import subprocess
import itertools
import numpy as np

argparser = argparse.ArgumentParser()

# argparser.add_argument("-main_path", help="path to main file", type=str)
# argparser.add_argument("-list_path", help="path to file containing list of jobs", type=str)
argparser.add_argument("-job_script_path", help="path to skeleton file for job instructions", type=str)
# argparser.add_argument("-num_cores", type=int, help="number of cores for job to use")

args = argparser.parse_args()

# list_of_jobs = []

# with open(args.list_path) as file:
#     jobs = [j.rstrip() for j in file.readlines() if j.rstrip()]
#     for job in jobs:
#         list_of_jobs.append(job)

job_combos = itertools.product(np.arange(4), np.linspace(0, 1, 11), np.linspace(0, 1, 11))

current_file_path = os.path.dirname(os.path.realpath(__file__))

for i, job_combo in enumerate(job_combos):
    job_file_copy = os.path.join(current_file_path, f'job_copy_{i}')
    print(job_file_copy)
    copyfile(args.job_script_path, job_file_copy)
    with open(job_file_copy, 'a') as file:
        file.write(f"python $HOME/sl_projects/student_teacher_catastrophic/run/run_pipeline.py --config $HOME/sl_projects/student_teacher_catastrophic/run/config.yaml -seed {job_combo[0]} -fa {job_combo[1]} -ra {job_combo[2]}\n")
        file.write("mkdir $WORK/$PBS_JOBID\n")
        file.write("cp * $WORK/$PBS_JOBID\n")
        subprocess.call(f"qsub job_copy_{i}", shell=True)
