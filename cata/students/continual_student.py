import torch
import torch.nn as nn
from cata.students import base_student


class ContinualStudent(base_student.BaseStudent):
    def _construct_output_layers(self):
        """Instantiate the output layers."""
        # create one head per teacher
        self._heads = nn.ModuleList([])
        for _ in range(self._num_teachers):
            output_layer = nn.Linear(
                self._layer_dimensions[-1], self._output_dimension, bias=self._bias
            )
            if self._soft_committee:
                output_layer.weight.data.fill_(1)
            else:
                if self._initialise_outputs:
                    self._initialise_weights(output_layer, 1)
                else:
                    nn.init.zeros_(output_layer.weight)
                    if self._bias:
                        nn.init.zeros_(output_layer.bias)
            # freeze heads (unfrozen when current task)
            for param in output_layer.parameters():
                param.requires_grad = False
            self._heads.append(output_layer)

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant head of student (depending on current teacher)."""
        y = self._heads[self._current_teacher](x)
        if self._apply_nonlinearity_on_output:
            y = self._nonlinear_function(y)
        if self._classification_output:
            y = self._threshold(y)
        return y

    def _signal_task_boundary(self, new_task: int) -> None:
        """Alert student to teacher change. Freeze previous head, unfreeze new head."""
        # freeze weights of head for previous task
        self._freeze_head(self._current_teacher)
        self._unfreeze_head(new_task)
        if self._copy_head_at_switch:
            # copy weights from old task head to new task head
            with torch.no_grad():
                self._heads[new_task].weight.copy_(
                    self._heads[self._current_teacher].weight
                )

        self._current_teacher = new_task

    def _freeze_head(self, head_index: int) -> None:
        """Freeze weights of head for task with index head index."""
        for param in self._heads[head_index].parameters():
            param.requires_grad = False

    def _unfreeze_head(self, head_index: int) -> None:
        """Unfreeze weights of head for task with index head index."""
        for param in self._heads[head_index].parameters():
            param.requires_grad = True
