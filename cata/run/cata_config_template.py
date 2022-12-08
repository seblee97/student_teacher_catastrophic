from config_manager import config_field
from config_manager import config_template

from cata import constants


class CataConfigTemplate:

    _ode_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.IMPLEMENTATION,
                types=[str],
                requirements=[lambda x: x in [constants.CPP, constants.PYTHON]],
            ),
            config_field.Field(
                name=constants.TIMESTEP,
                types=[float, int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.ODE_RUN],
        dependent_variables=[constants.ODE_SIMULATION],
        dependent_variables_required_values=[[True]],
    )

    _task_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.LABEL_TASK_BOUNDARIES, types=[bool]),
            config_field.Field(
                name=constants.LEARNER_CONFIGURATION,
                types=[str],
                requirements=[lambda x: x in [constants.META, constants.CONTINUAL]],
            ),
            config_field.Field(
                name=constants.TEACHER_CONFIGURATION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.FEATURE_ROTATION,
                        constants.READOUT_ROTATION,
                        constants.IDENTICAL,
                        constants.BOTH_ROTATION,
                        constants.NODE_SHARING,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.NUM_TEACHERS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.LOSS_TYPE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.REGRESSION,
                        constants.CLASSIFICATION,
                    ]
                ],
            ),
        ],
        level=[constants.TASK],
    )

    _consolidation_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TYPE,
                key=constants.CONSOLIDATION_TYPE,
                types=[type(None), str],
                requirements=[
                    lambda x: x is None
                    or x
                    in [
                        constants.EWC,
                        constants.QUADRATIC,
                        constants.SYNAPTIC_INTELLIGENCE,
                        constants.NODE_CONSOLIDATION,
                        constants.NODE_CONSOLIDATION_HESSIAN,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.IMPORTANCE,
                types=[int, float],
            ),
        ],
        level=[constants.TRAINING, constants.CONSOLIDATION],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TOTAL_TRAINING_STEPS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.TRAIN_BATCH_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.LEARNING_RATE,
                types=[float, int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.LOSS_FUNCTION,
                types=[str],
                requirements=[lambda x: x in [constants.MSE, constants.BCE]],
            ),
            config_field.Field(
                name=constants.SCALE_HEAD_LR,
                types=[bool],
            ),
            config_field.Field(
                name=constants.SCALE_HIDDEN_LR,
                types=[bool],
            ),
            config_field.Field(
                name=constants.TRAIN_HIDDEN_LAYERS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.TRAIN_HEAD_LAYER,
                types=[bool],
            ),
            config_field.Field(
                name=constants.FREEZE_FEATURES,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) for y in x)],
            ),
        ],
        nested_templates=[_consolidation_template],
        level=[constants.TRAINING],
    )

    _iid_gaussian_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MEAN,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.VARIANCE,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.DATASET_SIZE,
                types=[str, int],
                requirements=[lambda x: x == constants.INF or x > 0],
            ),
        ],
        level=[constants.DATA, constants.IID_GAUSSIAN],
        dependent_variables=[constants.INPUT_SOURCE],
        dependent_variables_required_values=[[constants.IID_GAUSSIAN]],
    )

    _data_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.INPUT_SOURCE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.IID_GAUSSIAN,
                        constants.MNIST_STREAM,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.NOISE_TO_STUDENT,
                types=[list],
                requirements=[
                    lambda x: all(
                        [
                            isinstance(y, list)
                            and all(
                                [isinstance(z, float) or isinstance(z, int) for z in y]
                            )
                            for y in x
                        ]
                    )
                ],
            ),
        ],
        nested_templates=[_iid_gaussian_template],
        level=[constants.DATA],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.VERBOSE,
                types=[bool],
            ),
            config_field.Field(
                name=constants.VERBOSE_TB,
                types=[int],
                requirements=[lambda x: x in [0, 1, 2]],
            ),
            config_field.Field(
                name=constants.LOG_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.CHECKPOINT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.PRINT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.SAVE_WEIGHT_FREQUENCY,
                types=[int, type(None)],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.LOG_TO_DF,
                types=[bool],
            ),
            config_field.Field(
                name=constants.MERGE_AT_CHECKPOINT,
                types=[bool],
            ),
            config_field.Field(
                name=constants.SAVE_WEIGHTS_AT_SWITCH,
                types=[bool],
            ),
            config_field.Field(
                name=constants.SAVE_INITIAL_WEIGHTS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.SAVE_TEACHER_WEIGHTS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.LOG_OVERLAPS,
                types=[bool],
            ),
            config_field.Field(name=constants.SPLIT_LOGGING, types=[bool]),
        ],
        level=[constants.LOGGING],
    )

    _testing_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TEST_BATCH_SIZE,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.TEST_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.OVERLAP_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.TESTING],
    )

    _student_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TEACHER_FEATURES_COPY,
                types=[type(None), int],
                requirements=[lambda x: x is None or x >= 0],
            ),
            config_field.Field(
                name=constants.STUDENT_HIDDEN_LAYERS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
            ),
            config_field.Field(
                name=constants.STUDENT_NONLINEARITY,
                types=[str],
                requirements=[
                    lambda x: x
                    in [constants.SCALED_ERF, constants.RELU, constants.LINEAR]
                ],
            ),
            config_field.Field(
                name=constants.APPLY_NONLINEARITY_ON_OUTPUT,
                types=[bool],
            ),
            config_field.Field(
                name=constants.STUDENT_INITIALISATION_STD,
                types=[float, int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.INITIALISE_STUDENT_OUTPUTS,
                types=[bool],
            ),
            config_field.Field(name=constants.COPY_HEAD_AT_SWITCH, types=[bool]),
            config_field.Field(
                name=constants.SOFT_COMMITTEE,
                types=[bool],
            ),
            config_field.Field(
                name=constants.STUDENT_BIAS_PARAMETERS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.SCALE_STUDENT_FORWARD_BY_HIDDEN,
                types=[bool],
            ),
            config_field.Field(
                name=constants.SYMMETRIC_STUDENT_INITIALISATION,
                types=[bool],
            ),
        ],
        level=[constants.MODEL, constants.STUDENT],
    )

    _feature_rotation_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.ROTATION_MAGNITUDE,
                key=constants.FEATURE_ROTATION_MAGNITUDE,
                types=[float, int],
            )
        ],
        level=[
            constants.MODEL,
            constants.TEACHERS,
            constants.FEATURE_ROTATION,
        ],
        dependent_variables=[constants.TEACHER_CONFIGURATION],
        dependent_variables_required_values=[[constants.FEATURE_ROTATION]],
    )

    _readout_rotation_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.ROTATION_MAGNITUDE,
                key=constants.READOUT_ROTATION_MAGNITUDE,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.FEATURE_COPY_PERCENTAGE,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 100],
            ),
        ],
        level=[
            constants.MODEL,
            constants.TEACHERS,
            constants.READOUT_ROTATION,
        ],
        dependent_variables=[constants.TEACHER_CONFIGURATION],
        dependent_variables_required_values=[[constants.READOUT_ROTATION]],
    )

    _both_rotation_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.READOUT_ROTATION_ALPHA,
                types=[float, int],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.FEATURE_ROTATION_ALPHA,
                types=[int, float],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
        ],
        level=[
            constants.MODEL,
            constants.TEACHERS,
            constants.BOTH_ROTATION,
        ],
        dependent_variables=[constants.TEACHER_CONFIGURATION],
        dependent_variables_required_values=[[constants.BOTH_ROTATION]],
    )

    _node_sharing_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.NUM_SHARED_NODES,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.ROTATION_MAGNITUDE,
                key=constants.FEATURE_ROTATION_MAGNITUDE,
                types=[float, int],
            ),
        ],
        level=[
            constants.MODEL,
            constants.TEACHERS,
            constants.NODE_SHARING,
        ],
        dependent_variables=[constants.TEACHER_CONFIGURATION],
        dependent_variables_required_values=[[constants.NODE_SHARING]],
    )

    _teachers_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TEACHER_NOISES,
                types=[list],
                requirements=[lambda x: all(y >= 0 for y in x)],
            ),
            config_field.Field(
                name=constants.TEACHER_HIDDEN_LAYERS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
            ),
            config_field.Field(
                name=constants.TEACHER_NONLINEARITIES,
                types=[list],
                requirements=[
                    lambda x: all(
                        y in [constants.SCALED_ERF, constants.RELU, constants.LINEAR]
                        for y in x
                    )
                ],
            ),
            config_field.Field(
                name=constants.NORMALISE_TEACHERS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.TEACHER_INITIALISATION_STD,
                types=[float, int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.UNIT_NORM_TEACHER_HEAD,
                types=[bool],
            ),
            config_field.Field(
                name=constants.TEACHER_BIAS_PARAMETERS,
                types=[bool],
            ),
            config_field.Field(
                name=constants.SCALE_TEACHER_FORWARD_BY_HIDDEN,
                types=[bool],
            ),
        ],
        nested_templates=[
            _feature_rotation_template,
            _readout_rotation_template,
            _both_rotation_template,
            _node_sharing_template,
        ],
        level=[constants.MODEL, constants.TEACHERS],
    )

    _model_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.INPUT_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.OUTPUT_DIMENSION,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        nested_templates=[_teachers_template, _student_template],
        level=[constants.MODEL],
    )

    _curriculum_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.STOPPING_CONDITION,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.FIXED_PERIOD,
                        constants.LOSS_THRESHOLDS,
                        constants.SWITCH_STEPS,
                    ]
                ],
            ),
            config_field.Field(
                name=constants.FIXED_PERIOD,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.SWITCH_STEPS,
                types=[list],
                requirements=[lambda x: all(isinstance(y, int) and y >= 0 for y in x)],
            ),
            config_field.Field(
                name=constants.LOSS_THRESHOLDS,
                types=[list],
                requirements=[
                    lambda x: all(
                        (isinstance(y, int) or isinstance(y, float)) and y > 0
                        for y in x
                    )
                ],
            ),
            config_field.Field(
                name=constants.INTERLEAVE_PERIOD, types=[int, type(None)]
            ),
            config_field.Field(
                name=constants.INTERLEAVE_DURATION, types=[int, type(None)]
            ),
        ],
        level=[constants.CURRICULUM],
    )

    base_config_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.EXPERIMENT_NAME,
                types=[str, type(None)],
            ),
            config_field.Field(
                name=constants.USE_GPU,
                types=[bool],
            ),
            config_field.Field(
                name=constants.GPU_ID,
                types=[int],
            ),
            config_field.Field(
                name=constants.SEED,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.RESULTS_PATH,
                types=[str, type(None)],
            ),
            config_field.Field(
                name=constants.NETWORK_SIMULATION,
                types=[bool],
            ),
            config_field.Field(
                name=constants.ODE_SIMULATION,
                types=[bool],
            ),
        ],
        nested_templates=[
            _ode_template,
            _task_template,
            _training_template,
            _data_template,
            _logging_template,
            _testing_template,
            _model_template,
            _curriculum_template,
        ],
    )
