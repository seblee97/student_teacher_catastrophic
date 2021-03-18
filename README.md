# Investigating Catastophic Forgetting with Student-Teacher Networks

This repository contains code for a project investigating catastrophic forgetting with 
a student-teacher framework. The project formed part of my thesis for an MSc in Computer Science at the University 
of Oxford, and was supervised by Andrew Saxe and Sebastian Goldt.

This code allows specification of a continual learning framework for a student
where different tasks are represented by different teachers.

# Prerequisites

To run this code you will need Python 3.8+ (other versions may work but have not been explicitly tested); it is written in PyTorch. All python package requirements are 
specified in the requirements file (requirements.txt) and can be satisfied by running (preferably in a virtual environment)

```pip3 install -r requirements.txt```

You will further need to add the root of this project to your python path variable. This can most easily be done by simply installing the repository as a package in your virtual environment
by running 

```pip install -e .```

from the root of the repository. Alternatively you can run

```PYTHONPATH="${PYTHONPATH}:$PATH_TO_ROOT"```

or place this line in your bash profile.

# Configurations

Experiment configurations can be found in the ```run/config.yaml``` file. 

# Experiments

Experiments are run with the ```python run/main.py``` command.