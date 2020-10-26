def positive_threshold(x):
    labels = torch.abs(y) > 0


def tanh_threshold(x):
    tanh_y = torch.tanh(y)
    labels = tanh_y > 0