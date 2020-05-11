# Investigating Catastophic Forgetting with Student-Teacher Networks

This repository contains code for a project investigating catastrophic forgetting with 
a student-teacher framework. The project is part of my thesis for an MSc in Computer Science at the University 
of Oxford, and is supervised by Andrew Saxe.

This code allows specification of a continual learning framework for a student
where different tasks are represented by different teachers.

# Prerequisites

To run this code you will need Python 3.7+; it is written in PyTorch. All other requirements are 
specified in the requirements file (requirements.txt) and can be satisfied by running  

```pip3 install -r requirements.txt```

# Teacher Configurations

noisy: each teacher has same initialisation, each with different levels of noise added
independent: all randomly initialised
drifting: all randomly initialised but drifting over time
overlapping: structured similarity in teacher weights

trained_mnist: teachers trained as classifiers

# Input Sources

iid_gaussian: drawn independently from Gaussian
mnist: inputs are flattened MNIST images