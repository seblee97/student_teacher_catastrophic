from models import Teacher, MetaStudent, ContinualStudent

from frameworks.teacher_student import StudentTeacher

import copy

class NoisyTeachers(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)

    def _setup_teacher_student_framework(self, config):
        # initialise student network
        task_setting = config.get(["task", "task_setting"])
        if task_setting == 'meta':
            self.student_network = MetaStudent(config=config).to(self.device)
        elif task_setting == 'continual':
            self.student_network = ContinualStudent(config=config).to(self.device)
        else:
            raise ValueError("Task setting {} not recognised. Please use 'meta' or 'continual'".format(task_setting))

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
    