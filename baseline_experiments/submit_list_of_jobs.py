import os
import argparse
from shutil import copyfile
import subprocess

argparser = argparse.ArgumentParser()

argparser.add_argument("-list_path", help="path to file containing list of jobs", type=str)
argparser.add_argument("-job_script_path", help="path to skeleton file for job instructions", type=str)

args = argparser.parse_args()

list_of_jobs = []

with open(args.list_path) as file:
    jobs = [j.rstrip() for j in file.readlines() if j.rstrip()]
    for job in jobs:
        list_of_jobs.append(job)

current_file_path = os.path.dirname(os.path.realpath(__file__))

for job in list_of_jobs:
    job_file_copy = os.path.join(current_file_path, 'job_copy.sh')
    print(job_file_copy)
    copyfile(args.job_script_path, job_file_copy)
    with open(job_file_copy, 'a') as file:
        file.write(job)
        subprocess.call("qsub -P saxe.prjc -q long.qc job_copy.sh", shell=True)
