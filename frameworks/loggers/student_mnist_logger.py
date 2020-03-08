from .base_logger import BaseLogger

from typing import Dict, List

class StudentMNISTLogger(BaseLogger):

    def __init__(self, config: Dict):
        BaseLogger.__init__(self, config)

    def _compute_layer_overlaps(self, student_network, teacher_networks: List):
        # extract layer weights
        if head is None:
            student_layer = self.student_network.state_dict()['layers.{}.weight'.format(layer)].data
            teacher_layers = [teacher.state_dict()['layers.{}.weight'.format(layer)].data for teacher in self.teachers]
        else:
            student_layer = self.student_network.state_dict()['heads.{}.weight'.format(str(head))].data
            teacher_layers = [teacher.state_dict()['output_layer.weight'].data for teacher in self.teachers]
            layer = layer + "_head_{}".format(str(head))

        # compute overlap matrices
        student_self_overlap = (student_layer.mm(student_layer.t()) / self.input_dimension).cpu().numpy()

        # log overlap values (scalars vs image graphs below)
        def log_matrix_values(log_name: str, matrix):
            matrix_shape = matrix.shape
            for i in range(matrix_shape[0]):
                for j in range(matrix_shape[1]):
                    if self.verbose_tb:
                        self.writer.add_scalar("layer_{}_{}/values_{}_{}".format(layer, log_name, i, j), matrix[i][j], step_count)
                    if self.log_to_df:
                        self.logger_df.at[step_count, "layer_{}_{}/values_{}_{}".format(layer, log_name, i, j)] = matrix[i][j]

        log_matrix_values("student_self_overlap", student_self_overlap)