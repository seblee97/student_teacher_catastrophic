import numpy as np
import torch


def linear_activation(x):
    return x


def scaled_erf_activation(x):
    return torch.erf(x / np.sqrt(2))
