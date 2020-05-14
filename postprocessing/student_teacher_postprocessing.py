from utils import Parameters

from typing import Dict

import os
import pandas as pd 
import yaml

class StudentTeacherPostprocessor:

    def __init__(self, save_path: str):

        self._save_path = save_path
        self.config = self._retrieve_config()

        self._logfile_path = self.config.get("logfile_path")

    def _retrieve_config(self):
         # read parameters from config in save path
        config_path = os.path.join(self._save_path, "config.yaml")
        with open(config_path, 'r') as yaml_file:
            params = yaml.load(yaml_file, yaml.SafeLoader)

        # create object in which to store experiment parameters and validate config file
        config_parameters = Parameters(params)
        
        return config_parameters

    def post_process(self) -> None:
        self._consolidate_dfs()

    def _consolidate_dfs(self):
        print("Consolidating/Merging all dataframes..")
        all_df_paths = [os.path.join(self._save_path, f) for f in os.listdir(self._save_path) if f.endswith('.csv')]
        ordered_df_paths = sorted(all_df_paths, key=lambda x: float(x.split("iter_")[-1].strip(".csv")))
        all_dfs = [pd.read_csv(df_path) for df_path in ordered_df_paths]
        merged_df = pd.concat(all_dfs)
        merged_df.to_csv(os.path.join(self._save_path, "data_logger.csv"))

        # remove indiviudal dataframes
        for df in all_df_paths:
            os.remove(df)
    