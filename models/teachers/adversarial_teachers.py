from .base_teachers import _BaseTeachers

import torch.optim as optim


class AdversarialTeachers(_BaseTeachers):

    def __init__(self, config):
        raise NotImplementedError(
            "Unsure how to supervise teacher i.e. should it know about \
            scores from student and be trained to maximise student error?"
            )
        _BaseTeachers.__init__(self, config)

        trainable_parameters = [
            filter(lambda param: param.requires_grad, teacher.parameters())
            for teacher in self._teachers
            ]
        self.teacher_optimisers = [
            optim.SGD(teacher_parameters, lr=self.learning_rate)
            for teacher_parameters in trainable_parameters
            ]

    def _setup_teachers(self, config):
        # initialise teacher networks, freeze
        teacher_noise = config.get(["task", "teacher_noise"])
        if type(teacher_noise) is int:
            teacher_noises = [teacher_noise for _ in range(self.num_teachers)]
        elif type(teacher_noise) is list:
            assert len(teacher_noise) == self.num_teachers, \
                "Provide one noise for each teacher. {} noises given, \
                    {} teachers specified".format(
                        len(teacher_noise), self._num_teachers
                        )
            teacher_noises = teacher_noise

        self.teachers = []
        for t in range(self._num_teachers):
            teacher = self._init_teacher(config=config, index=t)
            if teacher_noises[t] != 0:
                teacher.set_noise_distribution(mean=0, std=teacher_noises[t])
            self.teachers.append(teacher)

    def _signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def _signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass
