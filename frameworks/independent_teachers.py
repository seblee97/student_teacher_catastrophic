from models import Teacher, MetaStudent, ContinualStudent

from frameworks.teacher_student import StudentTeacher

class IndependentTeachers(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)

    def _setup_teacher_student_framework(self, config):
        # initialise student network
        self.student_network = MetaStudent(config=config).to(self.device)

        # initialise teacher networks, freeze
        teacher_noise = config.get(["task", "teacher_noise"])
        if type(teacher_noise) is int:
            teacher_noises = [teacher_noise for _ in range(self.num_teachers)]
        elif type(teacher_noise) is list:
            assert len(teacher_noise) == self.num_teachers, \
            "Provide one noise for each teacher. {} noises given, {} teachers specified".format(len(teacher_noise), self.num_teachers)
            teacher_noises = teacher_noise

        self.teachers = []
        for t in range(self.num_teachers):
            teacher = Teacher(config=config).to(self.device)
            teacher.freeze_weights()
            teacher.set_noise_distribution(mean=0, std=teacher_noises[t])
            self.teachers.append(teacher)
    