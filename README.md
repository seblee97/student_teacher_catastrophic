# catastrophic

Basic requirements can (hopefully) be met with

```pip3 install numpy matplotlib torch torchvision PyYAML tensorboardX```

More details to follow.

Run experiment with 

```python experiments/main.py```

View tensorboard logs at ```experiments/results/```

# Teacher Configurations

noisy: each teacher has same initialisation, each with different levels of noise added
independent: all randomly initialised
drifting: all randomly initialised but drifting over time
overlapping: structured similarity in teacher weights

trained_mnist: teachers trained as classifiers

# Input Sources

iid_gaussian: drawn independently from Gaussian
mnist: inputs are flattened MNIST images