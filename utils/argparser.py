from typing import Dict


class Argparser:

    @staticmethod
    def process_parser(parser):

        parser.add_argument(
            '-config', type=str,
            help='path to configuration file for student \
                teacher experiment', default='base_config.yaml'
            )
        parser.add_argument(
            '-gpu_id', type=int,
            help='id of gpu to use if more than 1 available',
            default=0
            )
        parser.add_argument(
            '-log_ext', action='store_false',
            help='whether to write evaluation \
                data to external file as well as tensorboard'
            )
        parser.add_argument(
            '-plot_config_path', '--pcp', type=str,
            help='path to json for plot \
                config', default="plot_configs/summary_plots.json"
        )
        parser.add_argument(
            '-auto_post_process', '--app', action='store_false',
            help='whether to \
                automatically go into post processing after training loop'
        )
        parser.add_argument(
            '-post_processing_path', '--ppp', type=str,
            help='path to folder to \
                post-process', default=None
            )

        parser.add_argument(
            '-seed', '--s', type=int,
            help='seed to use for packages with prng',
            default=None
            )
        parser.add_argument(
            '-learner_configuration', '--lc', type=str,
            help="meta or continual",
            default=None
            )
        parser.add_argument(
            '-teacher_configuration', '--tc', type=str,
            help="noisy or independent or mnist", default=None
            )
        parser.add_argument(
            '-input_source', '--inp_s', type=str,
            help="mnist or iid_gaussian",
            default=None)
        parser.add_argument(
            '-input_dim', '--id', type=int,
            help="input dimension to networks",
            default=None
            )
        parser.add_argument(
            '-num_teachers', '--nt', type=int, default=None
            )
        parser.add_argument(
            '-loss_type', '--lty', type=str, default=None
            )
        parser.add_argument(
            '-loss_function', '--lf', type=str, default=None
            )
        parser.add_argument(
            '-curriculum_type', '--ct', type=str, help="standard or custom",
            default=None
            )
        parser.add_argument(
            '-selection_type', '--st', type=str, help="random or cyclical",
            default=None
            )
        parser.add_argument(
            '-custom_curriculum', '--cc', type=str,
            help="order of teacher tasks", default=None
            )
        parser.add_argument(
            '-stopping_condition', '--sc', type=str, help="threshold or \
                fixed_period", default=None
            )
        parser.add_argument(
            '-fixed_period', '--fp', type=int,
            help="time between teacher change",
            default=None
            )
        parser.add_argument(
            '-loss_threshold', '--lt', type=str,
            help="how low loss for current \
                teacher goes before switching (used with threshold)",
            default=None
            )
        parser.add_argument(
            '-student_nonlinearity', '--snl', type=str,
            help="which non linearity \
                to use for student", default=None
            )
        parser.add_argument(
            '-teacher_nonlinearities', '--tnl', type=str,
            help="which non \
                linearity to use for teacher", default=None
            )
        parser.add_argument(
            '-teacher_hidden', '--th', type=str,
            help="dimension of hidden layer \
                in teacher", default=None
            )
        parser.add_argument(
            '-student_hidden', '--sh', type=str,
            help="dimension of hidden layer \
                in student", default=None
            )
        parser.add_argument(
            '-zero_student_overlap', '--zsho', action='store_true',
            help='manually ensure zero self-overlap in initialisation \
                of student'
        )
        parser.add_argument(
            '-zero_teacher_overlap', '--ztho', action='store_true',
            help='manually ensure zero self-overlap in initialisation \
                of teachers'
        )
        parser.add_argument(
            '-learning_rate', '--lr', type=float,
            help="base learning rate",
            default=None
            )
        parser.add_argument(
            '-total_steps', '--ts', type=int,
            help="total timesteps to run \
                algorithm", default=None
            )
        parser.add_argument(
            '-experiment_name', '--en', type=str,
            help="name to give to experiment", default=None
            )
        parser.add_argument(
            '-verbose', '--v', type=int,
            help="whether to display prints",
            default=None
            )
        parser.add_argument(
            '-tb_verbosity', '--tbv', type=int, choices=[0, 1, 2],
            help="how much to log to tensorboard 0: nothing, 1: basics,"
            "2: everything",
            default=None
            )
        parser.add_argument(
            '-checkpoint_path', '--cp', type=str, help="where to log results",
            default=None
            )
        parser.add_argument(
            '-checkpoint_frequency', '--cf', type=float,
            help="how often to log results", default=None)

        parser.add_argument(
            '-teacher_overlaps', '--to', type=str,
            help="per layer overlaps \
                between teachers. Must be in format '[30, 20, etc.]'",
            default=None
            )
        parser.add_argument(
            '-teacher_noises', '--tn', type=str,
            help="noise (in terms of std) \
                to be added to output of each teacher.", default=None
            )
        parser.add_argument(
            '-crop_start', type=float, help="For post-processing, where along \
                x domain to start the plot (as a percentage)", default=None
            )
        parser.add_argument(
            '-crop_end', type=float, help="For post-processing, where along \
                x domain to end the plot (as a percentage)", default=None
            )
        parser.add_argument(
            '-plot_figure_name', '--pfn', type=str, 
            help="Save name for post-processing figure", default=None
            )
        parser.add_argument(
            '-individual_plot_figures', '--ipf', action='store_true',
            help="Flag ensures plots are not combined and saved separately"
            )
        parser.add_argument(
            '-no_legend', '--nl', action='store_true',
            help="Flag ensures plots do not render the legend"
            )
        parser.add_argument(
            '-repeats', action='store_true',
            help="Flag states that save path is folder of repeats"
            )

        args = parser.parse_args()

        return args

    @staticmethod
    def update_config_with_parser(args, params: Dict):

        # update parameters with (optional) args given in command line
        if args.s:
            params["seed"] = args.s
        if args.lc:
            params["task"]["learner_configuration"] = args.lc
        if args.tc:
            params["task"]["teacher_configuration"] = args.tc
        if args.inp_s:
            params["training"]["input_source"] = args.inp_s
        if args.id:
            params["model"]["input_dimension"] = args.id
        if args.nt:
            params["task"]["num_teachers"] = args.nt
        if args.lty:
            params["task"]["loss_type"] = args.lty
        if args.lf:
            params["training"]["loss_function"] = args.lf
        if args.ct:
            params["curriculum"]["type"] = args.ct
        if args.st:
            params["curriculum"]["selection_type"] = args.st
        if args.sc:
            params["curriculum"]["stopping_condition"] = args.sc
        if args.fp:
            params["curriculum"]["fixed_period"] = args.fp
        if args.ts:
            params["training"]["total_training_steps"] = args.ts
        if args.en:
            params["experiment_name"] = args.en
        if args.lr:
            params["training"]["learning_rate"] = args.lr
        if args.cf:
            params["checkpoint_frequency"] = args.cf
        if args.v is not None:
            params["logging"]["verbose"] = bool(args.v)
        if args.tbv:
            params["logging"]["verbose_tb"] = args.tbv
        if args.zsho:
            params["model"]["student_zero_hidden_overlap"] = args.zsho
        if args.ztho:
            params["model"]["teacher_zero_hidden_overlap"] = args.ztho

        # update specific parameters with (optional) args given in command line
        if args.cc:
            custom = [
                int(op) for op in "".join(args.cc).strip('[]').split(',')
                ]
            params["curriculum"]["custom"] = custom
        if args.to:
            overlaps = [
                int(op) for op in "".join(args.to).strip('[]').split(',')
                ]
            params["teachers"]["overlap_percentages"] = overlaps
        if args.tn:
            noises = [
                float(op) for op in "".join(args.tn).strip('[]').split(',')
                ]
            params["teachers"]["teacher_noise"] = noises
        if args.snl:
            params["model"]["student_nonlinearity"] = args.snl
        if args.tnl:
            teacher_nonlinearities = [
                str(nl).strip()
                for nl in "".join(args.tnl).strip('[]').split(',')
                ]
            params["model"]["teacher_nonlinearities"] = teacher_nonlinearities
        if args.sh:
            student_hidden = [
                int(h) for h in "".join(args.sh).strip('[]').split(',')
                ]
            params["model"]["student_hidden_layers"] = student_hidden
        if args.th:
            teacher_hidden = [
                int(h) for h in "".join(args.th).strip('[]').split(',')
                ]
            params["model"]["teacher_hidden_layers"] = teacher_hidden
        if args.lt:
            if isinstance(args.lt, float):
                params["curriculum"]["loss_threshold"] = args.lt
            else:
                threshold_sequence = [
                    float(thr)
                    for thr in "".join(args.lt).strip('[]').split(',')
                ]
                params["curriculum"]["loss_threshold"] = threshold_sequence

        return params
