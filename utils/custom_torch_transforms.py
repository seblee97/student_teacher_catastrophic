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