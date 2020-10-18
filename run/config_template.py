from config_manager import config_field
from config_manager import config_template

import constants


class ConfigTemplate:

    _task_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.LABEL_TASK_BOUNDARIES, types=[bool]
            ),
            config_field.Field(
                name=constants.Constants.LEARNER_CONFIGURATION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [constants.Constants.META, constants.Constants.CONTINUAL]
                ],
            ),
            config_field.Field(
                name=constants.Constants.TEACHER_CONFIGURATION,
                types=[str],
                requirements=[lambda x: x in [constants.Constants.OVERLAPPING]],
            ),
            config_field.Field(
                name=constants.Constants.NUM_TEACHERS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.LOSS_TYPE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.Constants.REGRESSION,
                        constants.Constants.CLASSIFICATION,
                    ]
                ],
            ),
        ],
        level=[constants.Constants.TASK],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.TOTAL_TRAINING_STEPS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.TRAIN_BATCH_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.LEARNING_RATE,
                types=[float, int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.LOSS_FUNCTION,
                types=[str],
                requirements=[
                    lambda x: x in [constants.Constants.MSE, constants.Constants.BCE]
                ],
            ),
            config_field.Field(
                name=constants.Constants.SCALE_HEAD_LR,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.SCALE_HIDDEN_LR,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.ODE_TIMESTEP,
                types=[float, int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.TRAIN_FIRST_LAYER,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.TRAIN_HEAD_LAYER,
                types=[bool],
            ),
        ],
        level=[constants.Constants.TRAINING],
    )

    _data_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.INPUT_SOURCE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.Constants.IID_GAUSSIAN,
                        constants.Constants.MNIST_STREAM,
                    ]
                ],
            )
        ],
        level=[constants.Constants.DATA],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.VERBOSE,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.VERBOSE_TB,
                types=[int],
                requirements=[lambda x: x in [0, 1, 2]],
            ),
            config_field.Field(
                name=constants.Constants.CHECKPOINT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.LOG_TO_DF,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.MERGE_AT_CHECKPOINT,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.SAVE_WEIGHTS_AT_SWITCH,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.SAVE_INITIAL_WEIGHTS,
                types=[bool],
            ),
        ],
        level=[constants.Constants.LOGGING],
    )

    _testing_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.TEST_BATCH_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.TEST_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.OVERLAP_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.Constants.TESTING],
    )

    _model_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.INPUT_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.STUDENT_HIDDEN_LAYERS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
            ),
            config_field.Field(
                name=constants.Constants.TEACHER_HIDDEN_LAYERS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
            ),
            config_field.Field(
                name=constants.Constants.OUTPUT_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.STUDENT_NONLINEARITY,
                types=[str],
                requirements=[
                    lambda x: x
                    in [constants.Constants.SCALED_ERF, constants.Constants.RELU]
                ],
            ),
            config_field.Field(
                name=constants.Constants.TEACHER_NONLINEARITIES,
                types=[list],
                requirements=[
                    lambda x: all(
                        y in [constants.Constants.SCALED_ERF, constants.Constants.RELU]
                        for y in x
                    )
                ],
            ),
            config_field.Field(
                name=constants.Constants.NORMALISE_TEACHERS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.TEACHER_INITIALISATION_STD,
                types=[float, int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.STUDENT_INITIALISATION_STD,
                types=[float, int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.UNIT_NORM_TEACHER_HEAD,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.INITIALISE_STUDENT_OUTPUTS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.SOFT_COMMITTEE,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.TEACHER_BIAS_PARAMETERS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.STUDENT_BIAS_PARAMETERS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.SYMMETRIC_STUDENT_INITIALISATION,
                types=[bool],
            ),
        ],
        level=[constants.Constants.MODEL],
    )

    _curriculum_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.STOPPING_CONDITION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [constants.Constants.FIXED_PERIOD, constants.Constants.THRESHOLD]
                ],
            ),
            config_field.Field(
                name=constants.Constants.FIXED_PERIOD,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.Constants.LOSS_THRESHOLDS,
                types=[list],
                requirements=[
                    lambda x: all(
                        (isinstance(y, int) or isinstance(y, float)) and y > 0
                        for y in x
                    )
                ],
            ),
        ],
        level=[constants.Constants.CURRICULUM],
    )

    _teachers_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.OVERLAP_TYPE,
                types=[list],
                requirements=[
                    lambda x: all(
                        y in [constants.Constants.COPY, constants.Constants.ROTATION]
                        for y in x
                    )
                ],
            ),
            config_field.Field(
                name=constants.Constants.OVERLAP_ROTATIONS,
                types=[list],
                requirements=[
                    lambda x: all(
                        (
                            isinstance(y, float)
                            or isinstance(y, int)
                            or y == constants.Constants.NOT_APPLICABLE
                        )
                        for y in x
                    )
                ],
            ),
            config_field.Field(
                name=constants.Constants.OVERLAP_PERCENTAGES,
                types=[list],
                requirements=[
                    lambda x: all(
                        (
                            (isinstance(y, float) and y >= 0 and y <= 100)
                            or (isinstance(y, int) and y >= 0 and y <= 100)
                            or (y == constants.Constants.NOT_APPLICABLE)
                        )
                        for y in x
                    )
                ],
            ),
            config_field.Field(
                name=constants.Constants.TEACHER_NOISES,
                types=[list],
                requirements=[lambda x: all(y >= 0 for y in x)],
            ),
        ],
        level=[constants.Constants.TEACHERS],
    )

    base_config_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.Constants.EXPERIMENT_NAME,
                types=[str, type(None)],
            ),
            config_field.Field(
                name=constants.Constants.USE_GPU,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.SEED,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.Constants.NETWORK_SIMULATION,
                types=[bool],
            ),
            config_field.Field(
                name=constants.Constants.ODE_SIMULATION,
                types=[bool],
            ),
        ],
        nested_templates=[
            _task_template,
            _training_template,
            _data_template,
            _logging_template,
            _testing_template,
            _model_template,
            _curriculum_template,
            _teachers_template,
        ],
    )
