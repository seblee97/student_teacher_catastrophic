from frameworks.base_frameworks.meta_learning import MetaLearner
from frameworks.base_frameworks.continual_learning import ContinualLearner

from frameworks.base_frameworks.noisy_teachers import NoisyTeachers
from frameworks.base_frameworks.independent_teachers import IndependentTeachers

from typing import Dict

class MetaNoisy(NoisyTeachers, MetaLearner):

    def __init__(self, config: Dict):
        MetaLearner.__init__(self, config)
        NoisyTeachers.__init__(self, config)

class MetaIndependent(IndependentTeachers, MetaLearner):

    def __init__(self, config: Dict):
        MetaLearner.__init__(self, config)
        IndependentTeachers.__init__(self, config)

class ContinualNoisy(NoisyTeachers, ContinualLearner):

    def __init__(self, config: Dict):
        ContinualLearner.__init__(self, config)
        NoisyTeachers.__init__(self, config)

class ContinualIndependent(IndependentTeachers, ContinualLearner):

    def __init__(self, config: Dict):
        ContinualLearner.__init__(self, config)
        IndependentTeachers.__init__(self, config)

