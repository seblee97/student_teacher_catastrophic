from models import Teacher, MetaStudent, ContinualStudent

from frameworks.teacher_student import StudentTeacher

class DriftingTeachers(StudentTeacher):

    def __init__(self, config):
        StudentTeacher.__init__(self, config)

    def _setup_teacher_student_framework(self, config):
        raise NotImplementedError

    