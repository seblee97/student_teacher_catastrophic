from models import Teacher, MetaStudent, ContinualStudent

from frameworks import StudentTeacher

import torch 

import copy

class OverlappingTeachers(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)

    def _setup_teachers(self, config):
        """
        Instantiate all teachers
        
        Start with 'original' teacher. 
        The instantiate given set of teachers copying specified number of weights 
        from original teacher to new teachers. Weight duplicates can be in input to hidden,
        hidden to output or both.
        """
        # initialise teacher networks, freeze
        teacher_noise = config.get(["task", "teacher_noise"])
        if type(teacher_noise) is int:
            teacher_noises = [teacher_noise for _ in range(self.num_teachers)]
        elif type(teacher_noise) is list:
            assert len(teacher_noise) == self.num_teachers, \
            "Provide one noise for each teacher. {} noises given, {} teachers specified".format(len(teacher_noise), self.num_teachers)
            teacher_noises = teacher_noise

        overlap_percentages = config.get(["task", "overlap_percentages"])

        self.teachers = []
        original_teacher = Teacher(config=config).to(self.device)
        original_teacher.freeze_weights()
        if teacher_noises[0] != 0:
            original_teacher_output_std = original_teacher.get_output_statistics()
            original_teacher.set_noise_distribution(mean=0, std=teacher_noises[0] * original_teacher_output_std)
        self.teachers.append(original_teacher)
        for t in range(self.num_teachers - 1):
            teacher = Teacher(config=config).to(self.device)
            for l, layer in enumerate(teacher.state_dict()):
                layer_shape = teacher.state_dict()[layer].shape 
                assert len(layer_shape) == 2, "shape of layer tensor is not 2. Check consitency of layer construction with task."
                for row in range(layer_shape[0]):
                    overlapping_weights_dim = round(0.01 * overlap_percentages[l] * layer_shape[1])
                    overlapping_weights = copy.deepcopy(original_teacher.state_dict()[layer][row][:overlapping_weights_dim])
                    teacher.state_dict()[layer][row][:overlapping_weights_dim] = overlapping_weights
            teacher.freeze_weights()
            if teacher_noises[t + 1] != 0:
                teacher_output_std = teacher.get_output_statistics()
                teacher.set_noise_distribution(mean=0, std=teacher_noises[t + 1] * teacher_output_std)
            self.teachers.append(teacher)

    def _signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def _signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass

    