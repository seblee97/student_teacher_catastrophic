import torch

class CustomFlatten(object):
    """Flatten the image"""

    def __call__(self, image):

        img = torch.flatten(image)

        return img


class ToFloat(object):
    """Flatten the image"""

    def __call__(self, image):

        return torch.Tensor(image).type(torch.FloatTensor)

class Standardize(object):
    """Normalise (for tensor) input[channel] = (input[channel] - mean[channel]) / std[channel]"""

    def __init__(self, mean, std):
        self.mean = mean 
        self.std = std

        self.nan_mask = torch.Tensor([1 if ch != 0 else 0 for ch in std])
        self.std = torch.Tensor([ch if ch != 0 else 1 for ch in std])

    def __call__(self, tensor):
        return self.nan_mask * ((tensor - self.mean) / self.std)


class ApplyPCA(object):

    def __init__(self, pca_transform):
        self.pca_transform = pca_transform

    def __call__(self, image):

        # add dummy batch dimension
        unsqueezed_image = image.unsqueeze(0)

        # perform pca transform
        img = self.pca_transform.transform(unsqueezed_image)

        # remove dummy batch dimension
        reduced_image = img.squeeze()

        return reduced_image