class Constants:

    LABEL_TASK_BOUNDARIES = "label_task_boundaries"
    LEARNER_CONFIGURATION = "learner_configuration"
    CONTINUAL = "continual"
    META = "meta"
    TEACHER_CONFIGURATION = "teacher_configuration"
    OVERLAPPING = "overlapping"
    NUM_TEACHERS = "num_teachers"
    LOSS_TYPE = "loss_type"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TASK = "task"
    TOTAL_TRAINING_STEPS = "total_training_steps"
    TRAIN_BATCH_SIZE = "train_batch_size"
    LEARNING_RATE = "learning_rate"
    LOSS_FUNCTION = "loss_function"
    MSE = "mse"
    BCE = "bce"
    SCALE_HEAD_LR = "scale_head_lr"
    SCALE_HIDDEN_LR = "scale_hidden_lr"
    ODE_TIMESTEP = "ode_timestep"
    TRAIN_FIRST_LAYER = "train_first_layer"
    TRAIN_HEAD_LAYER = "train_head_layer"
    TRAINING = "training"
    INPUT_SOURCE = "input_source"
    IID_GAUSSIAN = "iid_gaussian"
    MNIST_STREAM = "mnist_stream"
    DATA = "data"
    VERBOSE = "verbose"
    VERBOSE_TB = "verbose_tb"
    CHECKPOINT_FREQUENCY = "checkpoint_frequency"
    LOG_TO_DF = "log_to_df"
    MERGE_AT_CHECKPOINT = "merge_at_checkpoint"
    SAVE_WEIGHTS_AT_SWITCH = "save_weights_at_switch"
    SAVE_INITIAL_WEIGHTS = "save_initial_weights"
    LOGGING = "logging"
    TEST_BATCH_SIZE = "test_batch_size"
    TEST_FREQUENCY = "test_frequency"
    OVERLAP_FREQUENCY = "overlap_frequency"
    TESTING = "testing"
    INPUT_DIMENSION = "input_dimension"
    STUDENT_HIDDEN_LAYERS = "student_hidden_layers"
    TEACHER_HIDDEN_LAYERS = "teacher_hidden_layers"
    OUTPUT_DIMENSION = "output_dimension"
    STUDENT_NONLINEARITY = "student_nonlinearity"
    SCALED_ERF = "scaled_erf"
    RELU = "relu"
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    TEACHER_NONLINEARITIES = "teacher_nonlinearities"
    NORMALISE_TEACHERS = "normalise_teachers"
    TEACHER_INITIALISATION_STD = "teacher_initialisation_std"
    STUDENT_INITIALISATION_STD = "student_initialisation_std"
    UNIT_NORM_TEACHER_HEAD = "unit_norm_teacher_head"
    INITIALISE_STUDENT_OUTPUTS = "initialise_student_outputs"
    SOFT_COMMITTEE = "soft_committee"
    TEACHER_BIAS_PARAMETERS = "teacher_bias_parameters"
    STUDENT_BIAS_PARAMETERS = "student_bias_parameters"
    SYMMETRIC_STUDENT_INITIALISATION = "symmetric_student_initialisation"
    MODEL = "model"
    STOPPING_CONDITION = "stopping_condition"
    FIXED_PERIOD = "fixed_period"
    THRESHOLD = "threshold"
    LOSS_THRESHOLDS = "loss_thresholds"
    CURRICULUM = "curriculum"
    OVERLAP_TYPE = "overlap_type"
    COPY = "copy"
    ROTATION = "rotation"
    OVERLAP_ROTATIONS = "overlap_rotations"
    NOT_APPLICABLE = "n/a"
    OVERLAP_PERCENTAGES = "overlap_percentages"
    TEACHER_NOISES = "teacher_noises"
    TEACHERS = "teachers"
    EXPERIMENT_NAME = "experiment_name"
    USE_GPU = "use_gpu"
    SEED = "seed"
    NETWORK_SIMULATION = "network_simulation"
    ODE_SIMULATION = "ode_simulation"

    CHECKPOINT_PATH = "checkpoint_path"
    EXPERIMENT_TIMESTAMP = "experiment_timestamp"
    RESULTS = "results"

    WEIGHT = "weight"

    EVEN_ODD_MAPPING = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1}

    GREATER_FIVE_MAPPING = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

    # Hard-coded subplot layouts for different numbers of graphs
    GRAPH_LAYOUTS = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (2, 4),
        8: (2, 4),
        9: (3, 3),
        10: (2, 5),
        11: (3, 4),
        12: (3, 4),
        13: (4, 4),
        14: (4, 4),
        15: (4, 4),
        16: (4, 4),
    }

    TEACHER_SHADES = ["#2A9D8F", "#E9C46A"]

    STUDENT_SHADES = ["#264653", "#E9C46A", "#878E88", "#76BED0"]

    ORANGE_SHADES = [
        "#E9C46A",
        "#F4A261",
        "#E76F51",
        "#D5B942",
        "#D9D375",
        "#EDFBC1",
        "#FC9E4F",
        "#F17105",
    ]

    TORQUOISE_SHADES = [
        "#2A9D8F",
        "#4E8098",
        "#17301C",
        "#4B644A",
        "#89A894",
        "#1C3738",
        "#32746D",
        "#01200F",
    ]

    BLUE_SHADES = ["#5465ff", "#788bff", "#9bb1ff", "#bfd7ff", "#e2fdff"]

    GREEN_SHADES = ["#143601", "#245501", "#538d22", "#73a942", "#aad576"]

    MNIST_TRAIN_SET_SIZE = 60000
    MNIST_TEST_SET_SIZE = 10000
    MNIST_FLATTENED_DIM = 784
