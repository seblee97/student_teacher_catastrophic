from cata.regularisers import base_regulariser
from torch import nn


class SynapticIntelligence(base_regulariser.BaseRegulariser):
    def compute_first_task_importance(
        self,
        student: nn.Module,
        previous_teacher_index: int,
        previous_teacher: nn.Module,
        loss_function,
        data_module,
    ):
        self._student = student

        self._params = {
            n: p for n, p in self._student.named_parameters() if "heads" not in n
        }
        self._store_previous_task_parameters()

        self._param_contributions = {
            n: p for n, p in student.previous_task_path_integral_contributions.items()
        }

    def penalty(self, student: nn.Module):
        loss = 0
        for n, param in student.named_parameters():
            if "heads" not in n:
                _loss = (
                    self._param_contributions[n]
                    * (param - self._previous_task_parameters[0][n]) ** 2
                )
                loss += _loss.sum()
        return self._importance * loss
