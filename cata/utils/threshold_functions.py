import torch

def positive_threshold(x):
    labels = torch.abs(x) > 0
    return labels


def tanh_threshold(x):
    tanh_y = torch.tanh(x)
    labels = tanh_y > 0
    return labels