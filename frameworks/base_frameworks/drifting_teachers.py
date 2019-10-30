from models import DriftingTeacher

from frameworks import StudentTeacher

class DriftingTeachers(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)
        self.drift_frequency = config.get(["task", "drift_frequency"])

    def _setup_teachers(self, config):
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
            teacher = DriftingTeacher(config=config).to(self.device)
            teacher.freeze_weights()
            teacher.set_noise_distribution(mean=0, std=teacher_noises[t])
            self.teachers.append(teacher)

    def _signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def _signal_step_boundary_to_teacher(self, step: int, current_task: int):
        self.teachers[current_task].forward_count += 1
        if step % self.drift_frequency == 0:
            self.teachers[current_task].rotate_weights()

    