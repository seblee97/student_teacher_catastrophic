from models import Teacher, MetaStudent, ContinualStudent

from frameworks import StudentTeacher

import copy

class NoisyTeachers(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)

    def _setup_teachers(self, config):
        # initialise teacher networks, freeze
        teacher_noises = config.get(["task", "teacher_noises"])

        assert len(teacher_noises) == self.num_teachers, \
        "Provide one noise for each teacher. {} noises given, {} teachers specified".format(len(teacher_noises), self.num_teachers)

        self.teachers = []
        base_teacher = Teacher(config=config).to(self.device)
        base_teacher.freeze_weights()
        for t in range(self.num_teachers):
            teacher = copy.deepcopy(base_teacher)
            teacher.set_noise_distribution(mean=0, std=teacher_noises[t])
            self.teachers.append(teacher)
    