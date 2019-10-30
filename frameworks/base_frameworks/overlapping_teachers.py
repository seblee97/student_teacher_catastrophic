from models import Teacher, MetaStudent, ContinualStudent

from frameworks.teacher_student import StudentTeacher

import copy

class OverlappingTeachers(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)

    def _setup_teachers(self, config):
        # initialise teacher networks, freeze
        teacher_noise = config.get(["task", "teacher_noise"])
        if type(teacher_noise) is int:
            teacher_noises = [teacher_noise for _ in range(self.num_teachers)]
        elif type(teacher_noise) is list:
            assert len(teacher_noise) == self.num_teachers, \
            "Provide one noise for each teacher. {} noises given, {} teachers specified".format(len(teacher_noise), self.num_teachers)
            teacher_noises = teacher_noise

        overlap_percentage = config.get(["task", "overlap_percentage"])

        self.teachers = []
        original_teacher = Teacher(config=config).to(self.device)
        original_teacher.freeze_weights()
        original_teacher.set_noise_distribution(mean=0, std=teacher_noises[0])
        self.teachers.append(original_teacher)
        for t in range(self.num_teachers - 1):
            teacher = copy.deepcopy(original_teacher)

            self.teachers.append(teacher)

    def _signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def _signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass

    