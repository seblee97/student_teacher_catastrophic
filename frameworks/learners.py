from frameworks.base_frameworks.meta_learning import MetaLearner
from frameworks.base_frameworks.continual_learning import ContinualLearner

from frameworks.base_frameworks.noisy_teachers import NoisyTeachers
from frameworks.base_frameworks.independent_teachers import IndependentTeachers
from frameworks.base_frameworks.drifting_teachers import DriftingTeachers
from frameworks.base_frameworks.overlapping_teachers import OverlappingTeachers
from frameworks.base_frameworks.mnist_reachers import MNISTTeachers

from typing import Dict

import torch
import torchvision

class MetaNoisy(NoisyTeachers, MetaLearner):

    def __init__(self, config: Dict):
        MetaLearner.__init__(self, config)
        NoisyTeachers.__init__(self, config)

class MetaIndependent(IndependentTeachers, MetaLearner):

    def __init__(self, config: Dict):
        MetaLearner.__init__(self, config)
        IndependentTeachers.__init__(self, config)

class MetaDrifting(DriftingTeachers, MetaLearner):

    def __init__(self, config: Dict):
        MetaLearner.__init__(self, config)
        DriftingTeachers.__init__(self, config)

class MetaOverlapping(OverlappingTeachers, MetaLearner):

    def __init__(self, config: Dict):
        MetaLearner.__init__(self, config)
        OverlappingTeachers.__init__(self, config)

class ContinualNoisy(NoisyTeachers, ContinualLearner):

    def __init__(self, config: Dict):
        ContinualLearner.__init__(self, config)
        NoisyTeachers.__init__(self, config)

class ContinualIndependent(IndependentTeachers, ContinualLearner):

    def __init__(self, config: Dict):
        ContinualLearner.__init__(self, config)
        IndependentTeachers.__init__(self, config)

class ContinualDrifting(DriftingTeachers, ContinualLearner):

    def __init__(self, config: Dict):
        ContinualLearner.__init__(self, config)
        DriftingTeachers.__init__(self, config)

class ContinualOverlapping(OverlappingTeachers, ContinualLearner):

    def __init__(self, config: Dict):
        ContinualLearner.__init__(self, config)
        OverlappingTeachers.__init__(self, config)

class ContinualMNIST:

    def __init__(self, config: Dict):
        self.data_path = config.get('data_path')

        # transforms to add to data
        transform = torchvision.transforms.Compose([
            transforms.ToTensor()
        ])

        mnist_data = torchvision.datasets.MNIST(self.data_path, transform=transform)
        data_loader = torchvision.
        raise NotImplementedError

class MetaMNIST(MNISTTeachers, MNISTMetaLearner):

    def __init__(self, config: Dict):
        raise NotImplementedError