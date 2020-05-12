from context import utils

from typing import List

class ModelTemplate(utils._Template):

    LEVELS = ["model"]
    OPTIONAL = []

    # Model level fields
    INPUT_DIMENSION = utils.Field(
        name="input_dimension", types=(int), reqs=[lambda x: x > 0]
    )

    STUDENT_HIDDEN_LAYERS = utils.Field(
        name="student_hidden_layers", types=(list), reqs=[lambda x: all(isinstance(y, int) and y > 0 for y in x)]
    )
    
    TEACHER_HIDDEN_LAYERS = utils.Field(
        name="teacher_hidden_layers", types=(list), reqs=[lambda x: all(isinstance(y, int) and y > 0 for y in x)]
    )

    OUTPUT_DIMENSION = utils.Field(
        name="output_dimension", types=(int), reqs=[lambda x: x > 0]
    )

    STUDENT_NONLINEARITY = utils.Field(
        name="student_nonlinearity", types=(str), reqs=[lambda x: x in ["relu", "linear", "sigmoid"]]
    )

    TEACHER_NONLINEARITIES = utils.Field(
        name="teacher_nonlinearities", types=(list), reqs=[lambda x: all(isinstance(y, str) and y in ["relu", "linear", "sigmoid"] for y in x)]
    )

    TEACHER_INITIALISATION_STD = utils.Field(
        name="teacher_initialisation_std", types=(float, int), reqs=[lambda x: x > 0]
    )

    STUDENT_INITIALISATION_STD = utils.Field(
        name="student_initialisation_std", types=(float, int), reqs=[lambda x: x > 0]
    )

    INITIALISE_STUDENT_OUTPUTS = utils.Field(
        name="initialise_student_outputs", types=(bool), reqs=None
    )

    SOFT_COMMITTEE = utils.Field(
        name="soft_committee", types=(bool), reqs=None
    )

    TEACHER_BIAS_PARAMETERS = utils.Field(
        name="teacher_bias_parameters", types=(bool), reqs=None
    )

    STUDENT_BIAS_PARAMETERS = utils.Field(
        name="student_bias_parameters", types=(bool), reqs=None
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.INPUT_DIMENSION,
            cls.STUDENT_HIDDEN_LAYERS,
            cls.TEACHER_HIDDEN_LAYERS,
            cls.OUTPUT_DIMENSION,
            cls.STUDENT_NONLINEARITY,
            cls.TEACHER_NONLINEARITIES,
            cls.TEACHER_INITIALISATION_STD,
            cls.STUDENT_INITIALISATION_STD,
            cls.INITIALISE_STUDENT_OUTPUTS,
            cls.SOFT_COMMITTEE,
            cls.TEACHER_BIAS_PARAMETERS,
            cls.STUDENT_BIAS_PARAMETERS
        ]