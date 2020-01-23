import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

import os
import warnings
import argparse

from PyPDF2 import PdfFileMerger

from typing import List, Tuple, Dict

parser = argparse.ArgumentParser()

parser.add_argument("-path_to_dir", type=str, help="path to directory of experimental results")
parser.add_argument("-attribute_list_path", type=str, help="path to file containing list of attributes to process")
parser.add_argument("-steps", type=int, default=100000)
parser.add_argument("-pdf_name", type=str)
parser.add_argument("-save_figs", action='store_false')
parser.add_argument("-multiple_experiments", action='store_true')
parser.add_argument("-delete_after_merge", action='store_false')

args = parser.parse_args()

def _concatenate_pdfs(pdf_directory: str, output_file_name: str, delete_individuals: bool):

    pdfs = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('pdf')]

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(open(pdf, 'rb'))

    with open("{}.pdf".format(output_file_name), "wb") as fout:
        merger.write(fout)

    for pdf in pdfs:
        os.remove(pdf)

def _read_events_file(events_file_path: str, attribute: str) -> List:
    """
    returns values in tensorboard event file associated with the given attribute tag
    :param events_file_path: path to tensorboard events file
    :param attribute: name of tag to filter events file for
    :return values: list of values associated with attibute.
    """
    values = []
    iterator = summary_iterator(events_file_path)
    while True:
        try:
            e = next(iterator)
            try:
                for v in e.summary.value:
                    if v.tag == attribute:
                        values.append(v.simple_value)
            except:
                pass
        except: 
            return values

    if len(values) == 0:
        raise Exception("Event file {} has no data for attribute {}".format(event_file_path, attribute))

    return values

def _make_plot(data: List[List[List]], labels: List[str], scale_axes: int=None, title: str=None, xlabel: str=None, ylabel: str=None, average: bool=True):
    """
    returns matplotlib figure of data provided
    :param data: list of list of list of values to plot. 
            Outer list is sets of data to plot. 
            Each set has repeats, each repeat has list of data.
    :param labels: label for each set of data
    :param scale_axes: in case readings are not taken every step for this data, scale x axis with this number as range.
    :param title: title for plot
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param average: whether to average lists or plot separately
    :return fig: matplotlib figure of plot
    """
    fig = plt.figure()
    for s, sub_data in enumerate(data):
        if average:
            if sum([len(s) == len(sub_data[0]) for s in sub_data]) != 0:
                warnings.warn("Note: averaging data over lists of different lengths. Longer lists will be truncated.\
                              Length of lists being averaged are: {}".format(str([len(s) for s in sub_data])), Warning)
                minimum_data_length = min([len(s) for s in sub_data])
                processed_sub_data = [s[:minimum_data_length] for s in sub_data]
            else:
                processed_sub_data = sub_data
            averaged_data = np.mean(processed_sub_data, axis=0)
            data_deviations = np.std(processed_sub_data, axis=0)

            if scale_axes:
                scaling = scale_axes / len(averaged_data)
                x_data = [i * scaling for i in range(len(averaged_data))]
            else:
                x_data = range(len(averaged_data))
            plt.plot(x_data, averaged_data, label=labels[s])
            plt.fill_between(
                x_data, averaged_data - data_deviations, averaged_data + data_deviations, alpha=0.3
            )
        else:
            for d, data_list in enumerate(sub_data):
                if scale_axes:
                    scaling = scale_axes / len(data_list)
                    x_data = [i * scaling for i in range(len(data_list))]
                else:
                    x_data = range(len(data_list))
                plt.plot(x_data, data_list, label="{}-repeat{}".format(labels[s], d))
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.5)
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
    if title:
        if average:
            plt.title("(average) {}".format(title))
        else:
            plt.title("(unaveraged) {}".format(title))
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    return fig

def smooth_values(values: List, window_width: int) -> List:
    """
    moving average of list of values
    :param values: raw values 
    :param window_width: width of moving average calculation
    :return smoothed_values: moving average values
    """
    cumulative_sum = np.cumsum(values, dtype=float)
    cumulative_sum[window_width:] = cumulative_sum[window_width:] - cumulative_sum[:-window_width]
    smoothed_values = cumulative_sum[window_width - 1:] / window_width
    return smoothed_values

class eventReader:
    
    def __init__(self, event_file_directory: str, multiple_experiments: bool=False):
        """
        :param event_file_directory: directory of event files. 
            Structure of directory should be as follows:
                parent:
                    |
                    - experiment_i
                        |
                        - seed_j
                            |
                            - event_file
                            |
                            - teacher_1
                                - event_file
                            - teacher_2
                                - event_file
                            
        """
        self.event_file_directory = event_file_directory
        
        # experiment names
        if multiple_experiments:
            self.experiment_folders = [f for f in os.listdir(self.event_file_directory) if not f.startswith(".") and f != 'figures']
        else:
            self.experiment_folders = [self.event_file_directory]

        # seeds
        if multiple_experiments:
            self.seed_directories = [[os.path.join(self.event_file_directory, e, f) for f in os.listdir(os.path.join(self.event_file_directory, e)) if not f.startswith(".")] for e in self.experiment_folders]
        else:
            self.seed_directories = [[os.path.join(self.event_file_directory, f) for f in os.listdir(e) if not f.startswith(".") and f != 'figures'] for e in self.experiment_folders]
        
        # teacher-agnostic event file paths sorted by experiment_name/seed
        self.general_event_file_paths = [[
            [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if 'tfevents' in f and os.path.getsize(os.path.join(sub_dir, f)) > 5] for sub_dir in seed_directory] for seed_directory in self.seed_directories
        ]
        
        # teacher-specific event file paths
        self.teacher1_event_file_paths = [[
            [os.path.join(sub_dir, 'teacher_0', f) for f in os.listdir(os.path.join(sub_dir, 'teacher_0')) if 'tfevents' in f and os.path.getsize(os.path.join(sub_dir, f)) > 5]
            for sub_dir in seed_directory] for seed_directory in self.seed_directories
        ]
        self.teacher2_event_file_paths = [[
            [os.path.join(sub_dir, 'teacher_1', f) for f in os.listdir(os.path.join(sub_dir, 'teacher_1')) if 'tfevents' in f and os.path.getsize(os.path.join(sub_dir, f)) > 5] 
            for sub_dir in seed_directory] for seed_directory in self.seed_directories
        ]

        
    def get_values(self, attributes: List[Tuple[str, bool]]):
        value_dict = {experiment: {} for experiment in self.experiment_folders}
        for attribute, general_attribute in attributes:
            for e, experiment_name in enumerate(self.experiment_folders):
                if general_attribute:
                    values = {"general": {"seed_{}".format(repeat): _read_events_file(event_file_path[0], attribute) for repeat, event_file_path in enumerate(self.general_event_file_paths[e])}}
                else:
                    teacher_1_values = {"teacher1": {"seed_{}".format(repeat): _read_events_file(event_file_path[0], attribute) for repeat, event_file_path in enumerate(self.teacher1_event_file_paths[e])}}
                    teacher_2_values = {"teacher2": {"seed_{}".format(repeat): _read_events_file(event_file_path[0], attribute) for repeat, event_file_path in enumerate(self.teacher2_event_file_paths[e])}}
                    values = dict(teacher_1_values, **teacher_2_values)
                value_dict[experiment_name][attribute] = values
        return value_dict
    

    def generate_plots(self, data: Dict, average=True, scale_axes=None):
        """
        :param data: nested dictionary of plot data.
            Structure of dictionary should be as follows:
                experiment_key: 
                    attrubute_key:
                        teacher_key:
                            seed_key:
                                data_value
        """
        all_plots = {}
        for experiment in data:
            for attribute in data[experiment]:
                if 'teacher1' in data[experiment][attribute]:
                    t1_all_seeds = list(data[experiment][attribute]['teacher1'].values())
                    t2_all_seeds = list(data[experiment][attribute]['teacher2'].values())
                    fig = _make_plot([t1_all_seeds, t2_all_seeds], average=average, title=attribute, labels=["Teacher 1", "Teacher 2"], scale_axes=scale_axes)
                    all_plots["{}_{}".format(experiment, attribute)] = fig
                elif 'general' in data[experiment][attribute]:
                    general_all_seeds = list(data[experiment][attribute]['general'].values())
                    fig = _make_plot([general_all_seeds], average=average, title=attribute, labels=[""], scale_axes=scale_axes)
                    all_plots["{}_{}".format(experiment, attribute)] = fig
        return all_plots
                    
        
if __name__ == "__main__":
    eR = eventReader(event_file_directory=args.path_to_dir, multiple_experiments=args.multiple_experiments)
    
    attributes = []
    with open(args.attribute_list_path) as f:
        for attribute_line in f:
            attribute_info = attribute_line.rstrip()
            attribute, general = attribute_info.split(",")
            attributes.append((attribute, int(general)))
            
    print(attributes)
    value_dict = eR.get_values(attributes)

    averaged_plots = eR.generate_plots(value_dict, average=True, scale_axes=args.steps)
    unaveraged_plots = eR.generate_plots(value_dict, average=False, scale_axes=args.steps)

    if args.save_figs:
        os.makedirs("{}/figures".format(args.path_to_dir), exist_ok=True)
        for figtitle in averaged_plots:
            figtitle_mod = figtitle.replace("/", "_")
            save_dir = os.path.join(args.path_to_dir, 'figures', 'average_{}'.format(figtitle_mod))
            averaged_plots[figtitle].savefig("{}.pdf".format(save_dir), dpi=1000)
        for figtitle in unaveraged_plots:
            figtitle_mod = figtitle.replace("/", "_")
            save_dir = os.path.join(args.path_to_dir, 'figures', figtitle_mod)
            unaveraged_plots[figtitle].savefig("{}.pdf".format(save_dir), dpi=1000)

    _concatenate_pdfs(
        os.path.join(args.path_to_dir, 'figures'), 
        output_file_name=os.path.join(args.path_to_dir, 'figures', args.pdf_name),
        delete_individuals=args.delete_after_merge 
        )
