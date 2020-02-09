import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import itertools

import os
import warnings
import argparse
import json
import subprocess

from PyPDF2 import PdfFileMerger

from typing import List, Tuple, Dict

parser = argparse.ArgumentParser()

parser.add_argument("-path_to_dir", type=str, help="path to directory of experimental results")
parser.add_argument("-attribute_list_path", type=str, help="path to file containing list of attributes to process")
parser.add_argument("-summary_plot_attributes", type=str, help="path to file containing list of attributes to plot for summary", default=None)
parser.add_argument("-animate", action='store_false')
parser.add_argument("-steps", type=int, default=100000)
parser.add_argument("-pdf_name", type=str)
parser.add_argument("-save_figs", action='store_false')
parser.add_argument("-multiple_experiments", action='store_true')
parser.add_argument("-delete_after_merge", action='store_false')
parser.add_argument("-start_percentile", type=int, default=0, help='percentile of the x range from which to start processing')
parser.add_argument("-end_percentile", type=int, default=100, help='percentile of the x range at which to stop processing')

args = parser.parse_args()

def _concatenate_pdfs(pdf_directory: str, output_file_name: str, delete_individuals: bool, keep_existing: List[str]):

    pdfs = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('pdf') and os.path.join(pdf_directory, f) not in keep_existing]

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(open(pdf, 'rb'))

    with open("{}.pdf".format(output_file_name), "wb") as fout:
        merger.write(fout)

    for pdf in pdfs:
        os.remove(pdf)

def _read_events_file(events_file_path: str, attribute: str, start: int=None, end: int=None) -> List:
    """
    returns values in tensorboard event file associated with the given attribute tag
    :param events_file_path: path to tensorboard events file
    :param attribute: name of tag to filter events file for
    (:param start: percentile of the x range from which to start reading)
    (:param end: percentile of the x range at which to stop reading)
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

def _make_plot(data: List[List[List]], labels: List[str], start: int=0, end: int=100, scale_axes: int=None, title: str=None, xlabel: str=None, ylabel: str=None, average: bool=True):
    """
    returns matplotlib figure of data provided
    :param data: list of list of list of values to plot. 
            Outer list is sets of data to plot. 
            Each set has repeats, each repeat has list of data.
    :param labels: label for each set of data
    :param start: percentile of the x range from which data processing started 
    :param end: percentile of the x range at which data processing ended 
    :param end: percentile of the x range at which to stop reading
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
                if len(averaged_data) == 0:
                    raise ArithmeticError("Division by zero. \
                        Please check attribute list to ensure data corresponding to each \
                        attribute exists in logs")
                scaling = scale_axes / len(averaged_data)
                x_data = [i * scaling for i in range(len(averaged_data))]

            else:
                x_data = range(len(averaged_data))
            
            # note currently inefficient since still requires going through whole iterator even if much is discarded. \
            # currently not an impediment so fine for now.
            # crop to specific region
            full_dataset_range = len(x_data)
            start_index = int(0.01 * start * full_dataset_range)
            end_index = int(0.01 * end * full_dataset_range)

            plt.plot(x_data[start_index: end_index], averaged_data[start_index: end_index], label=labels[s])
            plt.fill_between(
                x_data[start_index: end_index], (averaged_data - data_deviations)[start_index: end_index], (averaged_data + data_deviations)[start_index: end_index], alpha=0.3
            )
        else:
            for d, data_list in enumerate(sub_data):
                if scale_axes:
                    scaling = scale_axes / len(data_list)
                    x_data = [i * scaling for i in range(len(data_list))]
                else:
                    x_data = range(len(data_list))

                # crop to specific region
                full_dataset_range = len(x_data)
                start_index = int(0.01 * start * full_dataset_range)
                end_index = int(0.01 * end * full_dataset_range)

                plt.plot(x_data[start_index: end_index], data_list[start_index: end_index], label="{}-repeat{}".format(labels[s], d))
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
        for attribute, general_attribute, data_type in attributes:
            for e, experiment_name in enumerate(self.experiment_folders):
                if general_attribute:
                    values = {"general": {"seed_{}".format(repeat): _read_events_file(event_file_path[0], attribute) for repeat, event_file_path in enumerate(self.general_event_file_paths[e])}}
                else:
                    teacher_1_values = {"teacher1": {"seed_{}".format(repeat): _read_events_file(event_file_path[0], attribute) for repeat, event_file_path in enumerate(self.teacher1_event_file_paths[e])}}
                    teacher_2_values = {"teacher2": {"seed_{}".format(repeat): _read_events_file(event_file_path[0], attribute) for repeat, event_file_path in enumerate(self.teacher2_event_file_paths[e])}}
                    values = dict(teacher_1_values, **teacher_2_values)
                value_dict[experiment_name][attribute] = values
        return value_dict
    

    def generate_plots(self, data: Dict, average=True, scale_axes=None, start: int=0, end: int=100):
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
                    fig = _make_plot([t1_all_seeds, t2_all_seeds], average=average, title=attribute, labels=["Teacher 1", "Teacher 2"], scale_axes=scale_axes, start=start, end=end)
                    all_plots["{}_{}".format(experiment, attribute)] = fig
                elif 'general' in data[experiment][attribute]:
                    general_all_seeds = list(data[experiment][attribute]['general'].values())
                    fig = _make_plot([general_all_seeds], average=average, title=attribute, labels=[""], scale_axes=scale_axes, start=start, end=end)
                    all_plots["{}_{}".format(experiment, attribute)] = fig
        return all_plots

# Hard-coded subplot layouts for different numbers of graphs
LAYOUTS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (2, 4), 8: (2, 4), 9: (3, 3), 10: (2, 5), 11: (3, 4), 12: (3, 4)}

def summary_plot(data: Dict, path: str, scale_axes: int, start: float, end: float, figure_title: str):
    """
    Generate a 2x2 plot with 

        - generalisation error (log)
        - teacher-student overlap
        - student-student overlap
        - second layer weights

        or the relevant datasets specified by file at path.
    """
    # open json
    with open(path) as json_file:
        data_keys = json.load(json_file)

    figs = []

    number_of_graphs = len(data_keys.keys())

    rows = LAYOUTS[number_of_graphs][0]
    columns = LAYOUTS[number_of_graphs][1]

    width = 5
    height = 4

    heights = [height for _ in range(rows)]
    widths = [width for _ in range(columns)]

    for r in range(1):

        fig = plt.figure(constrained_layout=False, figsize=(columns * width, rows * height))

        spec = gridspec.GridSpec(nrows=rows, ncols=columns, width_ratios=widths, height_ratios=heights)

        key_index = 0

        for row in range(rows):
            for column in range(columns):

                if key_index < number_of_graphs:

                    fig_sub = fig.add_subplot(spec[row, column])

                    plot_index = list(data_keys.keys())[key_index]
                    i_data_keys = data_keys[plot_index]

                    for key in i_data_keys['keys']:
                        if i_data_keys["general"]:

                            general_y_data = data[key]['general']['seed_{}'.format(r)]

                            # scale axes
                            if scale_axes:
                                scaling = scale_axes / len(general_y_data)
                                x_data = [i * scaling for i in range(len(general_y_data))]
                            else:
                                x_data = range(len(general_y_data))
                            
                            # crop
                            full_dataset_range = len(x_data)
                            start_index = int(0.01 * start * full_dataset_range)
                            end_index = int(0.01 * end * full_dataset_range)
                            
                            # plot
                            fig_sub.plot(x_data[start_index: end_index], general_y_data[start_index: end_index])

                        else:

                            t1_y_data = data[key]['teacher1']['seed_{}'.format(r)]
                            t2_y_data = data[key]['teacher2']['seed_{}'.format(r)]

                            # scale axes
                            if scale_axes:
                                scaling = scale_axes / len(t1_y_data)
                                x_data = [i * scaling for i in range(len(t1_y_data))]
                            else:
                                x_data = range(len(t1_y_data))
                            
                            # crop
                            full_dataset_range = len(x_data)
                            start_index = int(0.01 * start * full_dataset_range)
                            end_index = int(0.01 * end * full_dataset_range)
                            
                            # plot
                            fig_sub.plot(x_data[start_index: end_index], t1_y_data[start_index: end_index], label='Teacher 1')
                            fig_sub.plot(x_data[start_index: end_index], t2_y_data[start_index: end_index], label='Teacher 2')

                        # labelling
                        fig_sub.set_xlabel("Step")
                        fig_sub.set_ylabel(i_data_keys["title"])
                        # column.set_xticklabels(["{:.3e}".format(t) for t in column.get_xticks()])
                        fig_sub.legend()

                        # grids
                        fig_sub.minorticks_on()
                        fig_sub.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.5)
                        fig_sub.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
                    
                    key_index += 1
        
        fig.suptitle("Summary Plot: {}".format(figure_title))

        figs.append(fig)

    return figs

def summary_plot_animate(data: Dict, path: str, scale_axes: int, start: float, end: float, figure_title: str, save_path: str): 

    # open json
    with open(path) as json_file:
        data_keys = json.load(json_file)

    figs = []

    number_of_graphs = len(data_keys.keys())

    rows = LAYOUTS[number_of_graphs][0]
    columns = LAYOUTS[number_of_graphs][1]

    width = 5
    height = 4

    heights = [height for _ in range(rows)]
    widths = [width for _ in range(columns)]

    for r in range(1):

        step_list = list(range(0, scale_axes, 250))

        for s, step in enumerate(step_list):

            fig = plt.figure(constrained_layout=False, figsize=(columns * width, rows * height))

            spec = gridspec.GridSpec(nrows=rows, ncols=columns, width_ratios=widths, height_ratios=heights)

            key_index = 0

            for row in range(rows):
                for column in range(columns):

                    if key_index < number_of_graphs:

                        fig_sub = fig.add_subplot(spec[row, column])

                        plot_index = list(data_keys.keys())[key_index]
                        i_data_keys = data_keys[plot_index]

                        if i_data_keys['image']:

                            # import pdb; pdb.set_trace()
                            
                            key_body_prefix, teacher_index, im_dim = i_data_keys['keys'].split('/')
                            key_body_suffix, x, y = im_dim.split('_')
                            x, y = int(x), int(y)
                            xy_tuples = list(itertools.product(list(range(x)), list(range(y))))

                            all_keys = {(xi, yi): "{}/{}/{}_{}_{}".format(key_body_prefix, teacher_index, key_body_suffix, xi, yi) 
                                        for (xi, yi) in xy_tuples}

                            matrix = np.zeros((x, y))

                            for xi in range(x):
                                for yi in range(y):
                                    general_y_data = data[all_keys[(xi, yi)]]['general']['seed_{}'.format(r)]
                                    num_data_points = len(general_y_data)

                                    cutoff = int(num_data_points * (s / len(step_list)))
                                    # if cutoff == len(general_y_data):
                                    #     cutoff -= 1
                        
                                    matrix[xi][yi] = general_y_data[cutoff]

                            im = fig_sub.imshow(matrix, vmin=0, vmax=1) 

                            # colorbar
                            divider = make_axes_locatable(fig_sub)
                            cax = divider.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(im, cax=cax, orientation='vertical')

                            # title and ticks
                            fig_sub.set_ylabel(i_data_keys["title"])
                            fig_sub.set_xticks([])
                            fig_sub.set_yticks([])
                        
                        else:
                            
                            for key in i_data_keys['keys']:

                                if i_data_keys["general"]:

                                    general_y_data = data[key]['general']['seed_{}'.format(r)]
                                    num_data_points = len(general_y_data)

                                    # scale axes
                                    if scale_axes:
                                        scaling = scale_axes / len(general_y_data)
                                        x_data = [i * scaling for i in range(len(general_y_data))]
                                    else:
                                        x_data = range(len(general_y_data))
                                    
                                    # crop
                                    full_dataset_range = len(x_data)
                                    start_index = int(0.01 * start * full_dataset_range)
                                    end_index = int(0.01 * end * full_dataset_range)
                                    
                                    # plot
                                    cutoff = int(num_data_points * (s / len(step_list)))
                                    fig_sub.plot(x_data[start_index: end_index][:cutoff], general_y_data[start_index: end_index][:cutoff])

                                else:

                                    t1_y_data = data[key]['teacher1']['seed_{}'.format(r)]
                                    t2_y_data = data[key]['teacher2']['seed_{}'.format(r)]

                                    num_data_points = len(t1_y_data)

                                    # scale axes
                                    if scale_axes:
                                        scaling = scale_axes / len(t1_y_data)
                                        x_data = [i * scaling for i in range(len(t1_y_data))]
                                    else:
                                        x_data = range(len(t1_y_data))
                                    
                                    # crop
                                    full_dataset_range = len(x_data)
                                    start_index = int(0.01 * start * full_dataset_range)
                                    end_index = int(0.01 * end * full_dataset_range)
                                    
                                    # plot
                                    cutoff = int(num_data_points * (s / len(step_list)))
                                    fig_sub.plot(x_data[start_index: end_index][:cutoff], t1_y_data[start_index: end_index][:cutoff], label='Teacher 1')
                                    fig_sub.plot(x_data[start_index: end_index][:cutoff], t2_y_data[start_index: end_index][:cutoff], label='Teacher 2')

                                    fig_sub.legend()

                                # labelling
                                fig_sub.set_xlabel("Step")
                                fig_sub.set_ylabel(i_data_keys["title"])
                                # column.set_xticklabels(["{:.3e}".format(t) for t in column.get_xticks()])

                                # grids
                                fig_sub.minorticks_on()
                                fig_sub.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.5)
                                fig_sub.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)

                                # limits
                                fig_sub.set_xlim([0, scale_axes])
                                fig_sub.set_ylim(i_data_keys["ylimits"])
                        
                        key_index += 1
            
            fig.suptitle("Summary Plot: {}".format(figure_title))

            fig.savefig("{}/image{}.png".format(save_path, f"{s:06d}"), dpi=500)
            plt.close()

def make_mp4(image_dir: str):

    os.chdir(image_dir)

    subprocess.call(["ffmpeg", "-f", "image2", "-r", "5", "-i", "image%06d.png", "-vcodec", "mpeg4", "-y", "summary_plot.mp4"])

    pngs = [f for f in os.listdir(os.getcwd()) if f.endswith('png')]

    for png in pngs:
        os.remove(png)
        
if __name__ == "__main__":
    eR = eventReader(event_file_directory=args.path_to_dir, multiple_experiments=args.multiple_experiments)
    
    attributes = []
    with open(args.attribute_list_path) as json_file:
        attribute_info = json.load(json_file)
        for key in attribute_info.keys():
            attributes.append((key, attribute_info[key]["general"], attribute_info[key]["type"]))

    # with open(args.attribute_list_path) as f:
    #     for attribute_line in f:
    #         attribute_info = attribute_line.rstrip()
    #         attribute, general = attribute_info.split(",")
    #         attributes.append((attribute, int(general)))
            
    print(attributes)
    value_dict = eR.get_values(attributes)

    if args.save_figs:
        os.makedirs("{}/figures".format(args.path_to_dir), exist_ok=True)

    if args.summary_plot_attributes:
        if args.animate:
            os.makedirs("{}/figures/animate".format(args.path_to_dir), exist_ok=True)
            summary_plots = summary_plot_animate(
                value_dict[list(value_dict.keys())[0]], args.summary_plot_attributes, scale_axes=args.steps, 
                start=args.start_percentile, end=args.end_percentile, figure_title=list(value_dict.keys())[0],
                save_path=os.path.join(args.path_to_dir, 'figures', 'animate')
                )
        else:
            summary_plots = summary_plot(
                value_dict[list(value_dict.keys())[0]], args.summary_plot_attributes, scale_axes=args.steps, 
                start=args.start_percentile, end=args.end_percentile, figure_title=list(value_dict.keys())[0]
                )

    averaged_plots = eR.generate_plots(value_dict, average=True, scale_axes=args.steps, start=args.start_percentile, end=args.end_percentile)
    unaveraged_plots = eR.generate_plots(value_dict, average=False, scale_axes=args.steps, start=args.start_percentile, end=args.end_percentile)

    if args.save_figs:

        figure_path = os.path.join(args.path_to_dir, 'figures')
        existing_pdfs = [os.path.join(figure_path, f) for f in os.listdir(figure_path) if f.endswith('pdf')]

        for figtitle in averaged_plots:
            figtitle_mod = figtitle.replace("/", "_")
            save_dir = os.path.join(args.path_to_dir, 'figures', 'average_{}'.format(figtitle_mod))
            averaged_plots[figtitle].savefig("{}.pdf".format(save_dir), dpi=1000)
        for figtitle in unaveraged_plots:
            figtitle_mod = figtitle.replace("/", "_")
            save_dir = os.path.join(args.path_to_dir, 'figures', figtitle_mod)
            unaveraged_plots[figtitle].savefig("{}.pdf".format(save_dir), dpi=1000)

        cwd = os.getcwd()

        if summary_plots:
            for r, fig in enumerate(summary_plots):
                save_dir = os.path.join(args.path_to_dir, 'figures', 'summary_plot_{}.pdf'.format(r))
                fig.savefig(save_dir, dpi=1000)
                existing_pdfs.append(save_dir)
        else:
            make_mp4(os.path.join(args.path_to_dir, 'figures', 'animate'))

        os.chdir(cwd)

        _concatenate_pdfs(
            os.path.join(args.path_to_dir, 'figures'), 
            output_file_name=os.path.join(args.path_to_dir, 'figures', args.pdf_name),
            delete_individuals=args.delete_after_merge,
            keep_existing=existing_pdfs
            )
