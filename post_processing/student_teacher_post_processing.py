import os
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

from post_processing.df_plotter import DataFramePlotter
from post_processing.weight_plotter import WeightPlotter
from utils import Parameters


class StudentTeacherPostProcessor:

    def __init__(self, save_path: str, extra_args: Dict):
        self._repeats = extra_args["repeats"]
        self._save_path = save_path

        config = self._retrieve_config()
        config = self._update_config(config=config, extra_args=extra_args)

        self._extract_parameters(config)

        # setup df plotter
        self.df_plotter = DataFramePlotter(config)

        if self._save_weights_at_switch:
            # setup weight plotter
            self.weight_plotter = WeightPlotter(config)

    def _retrieve_config(self):
        # read parameters from config in save path
        if self._repeats:
            config_paths = [
                os.path.join(self._save_path, f, "config.yaml")
                for f in os.listdir(self._save_path)
                if f != 'figures' and f != '.DS_Store'
            ]
            configs = []
            for config_path in config_paths:
                with open(config_path, 'r') as yaml_file:
                    params = yaml.load(yaml_file, yaml.SafeLoader)
                    configs.append(params)
            # TODO assertion about identical configs
            config_parameters = Parameters(configs[0])

        else:
            config_path = os.path.join(self._save_path, "config.yaml")
            with open(config_path, 'r') as yaml_file:
                params = yaml.load(yaml_file, yaml.SafeLoader)

            # create object in which to store experiment parameters and
            # validate config file
            config_parameters = Parameters(params)

        return config_parameters

    def _update_config(self, config: Parameters, extra_args: Dict) -> Parameters:

        config.set_property("repeats", self._repeats)
        config.set_property("save_path", self._save_path)

        previously_specified_crop_x = \
            config.get(["post_processing", "crop_x"])

        if previously_specified_crop_x is None:
            crop_x = [0, 1]
        else:
            crop_x = previously_specified_crop_x

        if extra_args.get("crop_start"):
            crop_x[0] = extra_args.get("crop_start")
        if extra_args.get("crop_end"):
            crop_x[1] = extra_args.get("crop_end")

        config._config["post_processing"]["crop_x"] = crop_x

        config._config["post_processing"]["combine_plots"] = \
            extra_args.get("combine_plots")
        config._config["post_processing"]["show_legends"] = \
            extra_args.get("show_legends")

        if extra_args.get("figure_name"):
            df_figure_name = extra_args.get("figure_name")
        else:
            if config.get(["post_processing", "combine_plots"]):
                df_figure_name = "summary_plot"
            else:
                df_figure_name = "individual_plot"
        config.set_property("df_figure_name", df_figure_name)

        return config

    def _extract_parameters(self, config: Parameters):
        self._save_weights_at_switch = \
            config.get(["logging", "save_weights_at_switch"])

    def post_process(self) -> None:

        self._consolidate_dfs()

        self.df_plotter.make_summary_plot()

        if self._save_weights_at_switch:
            self.weight_plotter.make_per_layer_weight_plot(plot_type="weight_pdf")
            self.weight_plotter.make_per_layer_weight_plot(plot_type="weight_diffs")
            self.weight_plotter.make_per_layer_weight_plot(plot_type="weight_diff_pdf")

    def _consolidate_dfs(self):
        if not self._repeats:
            print("Consolidating/Merging all dataframes..")
            all_df_paths = [
                os.path.join(self._save_path, f)
                for f in os.listdir(self._save_path)
                if f.endswith('.csv')
            ]

            if any("data_logger.csv" in path for path in all_df_paths):
                print("'data_logger.csv' file already in save path "
                      "specified. Consolidation already complete.")
                if len(all_df_paths) > 1:
                    print("Note, other csv files are also still in save path.")

            else:
                ordered_df_paths = sorted(
                    all_df_paths, key=lambda x: float(x.split("iter_")[-1].strip(".csv")))

                print("Loading all dataframes..")
                all_dfs = [pd.read_csv(df_path) for df_path in ordered_df_paths]
                print("Dataframes loaded. Merging..")
                merged_df = pd.concat(all_dfs)

                key_set = set()
                for df in all_dfs:
                    key_set.update(df.keys())

                assert set(merged_df.keys()) == key_set, \
                    "Merged df does not have correct keys"

                print("Dataframes merged. Saving...")
                merged_df.to_csv(os.path.join(self._save_path, "data_logger.csv"))

                print("Saved. Removing individual dataframes...")
                # remove individual dataframes
                for df in all_df_paths:
                    os.remove(df)
                print("Consolidation complete.")

    def _average_datasets(self, data: List[List[List[float]]]) -> Tuple[List[List[float]]]:
        assert all(
            len(rep) == len(data[0])
            for rep in data), "Varying number of data attributes are provided for different repeats"

        averaged_data = []
        data_deviations = []

        for attribute_index in range(len(data[0])):
            attribute_repeats = [data[r][attribute_index] for r in range(len(data))]
            maximum_data_length = max([len(dataset) for dataset in attribute_repeats])
            processed_sub_data = [
                np.pad(dataset, (0, maximum_data_length - len(dataset)), 'constant')
                for dataset in attribute_repeats
            ]

            nonzero_count = np.count_nonzero(processed_sub_data, axis=0)
            nonzero_masks = [(data != 0).astype(int) for data in processed_sub_data]
            attribute_sum = np.sum(processed_sub_data, axis=0)

            attribute_averaged_data = attribute_sum / nonzero_count

            attribute_differences = [
                mask * (attribute_averaged_data - data)**2
                for mask, data in zip(nonzero_masks, processed_sub_data)
            ]

            attribute_data_deviations = np.sqrt(np.sum(attribute_differences,
                                                       axis=0)) / nonzero_count

            averaged_data.append(attribute_averaged_data)
            data_deviations.append(attribute_data_deviations)

        return averaged_data, data_deviations
