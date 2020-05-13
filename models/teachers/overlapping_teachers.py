from .base_teachers import _BaseTeachers

import torch 
import copy

from typing import Dict

class OverlappingTeachers(_BaseTeachers):

    def __init__(self, config: Dict):
        _BaseTeachers.__init__(self, config)

    def _setup_teachers(self, config: Dict) -> None:
        """
        Instantiate all teachers.
        
        Start with 'original' teacher. 
        Then instantiate set of teachers by copying specified number of weights 
        from original teacher to new teachers. Weight duplicates can be in input to hidden,
        hidden to output or both.
        """
        # get noise configs if applicable
        teacher_noise = config.get(["teachers", "teacher_noise"])
        if type(teacher_noise) is int:
            teacher_noises = [teacher_noise for _ in range(self._num_teachers)]
        elif type(teacher_noise) is list:
            assert len(teacher_noise) == self._num_teachers, \
            "Provide one noise for each teacher. {} noises given, {} teachers specified".format(len(teacher_noise), self._num_teachers)
            teacher_noises = teacher_noise

        overlap_percentages = config.get(["teachers", "overlap_percentages"])

        self._teachers = []

        original_teacher = self._init_teacher(config=config, index=0)
        original_teacher.freeze_weights()

        # set noise for 'original' teacher
        if teacher_noises[0] != 0:
            original_teacher_output_std = original_teacher.get_output_statistics()
            original_teacher.set_noise_distribution(mean=0, std=teacher_noises[0] * original_teacher_output_std)

        self._teachers.append(original_teacher)

        # setup remaining teachers
        for t in range(self._num_teachers - 1):

            teacher = self._init_teacher(config=config, index=t + 1)

            for l, layer in enumerate(teacher.state_dict()):
                layer_shape = teacher.state_dict()[layer].shape 
                assert len(layer_shape) == 2, "shape of layer tensor is not 2. Check consitency of layer construction with task."
                for row in range(layer_shape[0]):
                    overlapping_weights_dim = round(0.01 * overlap_percentages[l] * layer_shape[1])
                    overlapping_weights = copy.deepcopy(original_teacher.state_dict()[layer][row][:overlapping_weights_dim])
                    teacher.state_dict()[layer][row][:overlapping_weights_dim] = overlapping_weights

            # freeze weights of every teacher
            teacher.freeze_weights()

            # set noise 
            if teacher_noises[t + 1] != 0:
                teacher_output_std = teacher.get_output_statistics()
                teacher.set_noise_distribution(mean=0, std=teacher_noises[t + 1] * teacher_output_std)
            self._teachers.append(teacher)

    def test_set_forward(self, teacher_index: int, batch: Dict) -> torch.Tensor:
        return self.forward(teacher_index, batch)

    def forward(self, teacher_index: int, batch: Dict) -> torch.Tensor:
        x = batch['x']
        output = self._teachers[teacher_index](x)
        return output

    def signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass

    