import pandas as pd 
import numpy as np 
import itertools
import copy

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation 
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse 
import os 
import json

from typing import Dict, List

parser = argparse.ArgumentParser()

parser.add_argument("-path_to_csv", type=str, help="path to dataframe with experimental results")
parser.add_argument("-summary_plot_attributes", type=str, help="path to file containing list of attributes to plot for summary", default=None)
parser.add_argument("-save_path", type=str, help="path to save figures")
parser.add_argument("-exp_name", type=str, help="name of experiment")

args = parser.parse_args()

# Hard-coded subplot layouts for different numbers of graphs
LAYOUTS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (2, 4), 8: (2, 4), 9: (3, 3), 10: (2, 5), 11: (3, 4), 12: (3, 4)}

class SummaryPlot:

    def __init__(self, data_csv, plot_config_path: str, save_path:str, experiment_name:str):

        self.data = pd.read_csv(data_csv)

        with open(plot_config_path) as json_file:
            self.plot_config = json.load(json_file)

        self.save_path = save_path
        self.experiment_name = experiment_name

        self.plot_keys = list(self.plot_config["data"].keys())

        self.number_of_graphs = len(self.plot_keys)

        self.rows = LAYOUTS[self.number_of_graphs][0]
        self.columns = LAYOUTS[self.number_of_graphs][1]

        width = self.plot_config['config']['width']
        height = self.plot_config['config']['height']

        heights = [height for _ in range(self.rows)]
        widths = [width for _ in range(self.columns)]

        self.fig = plt.figure(constrained_layout=False, figsize=(self.columns * width, self.rows * height))

        self.spec = gridspec.GridSpec(nrows=self.rows, ncols=self.columns, width_ratios=widths, height_ratios=heights)

    def add_subplot(self, plot_data, row_index: int, column_index: int, title: str, labels: List, ylimits: List, scale_axes: int):

        if len(labels) > 10:
            linewidth = 0.05
        else:
            linewidth = 1

        fig_sub = self.fig.add_subplot(self.spec[row_index, column_index])

        for d, dataset in enumerate(plot_data):
            # scale axes
            if scale_axes:
                scaling = scale_axes / len(dataset)
                x_data = [i * scaling for i in range(len(dataset))]
            else:
                x_data = range(len(dataset))

            fig_sub.plot(x_data, dataset, label=labels[d], linewidth=linewidth)
        
        # labelling
        fig_sub.set_xlabel("Step")
        fig_sub.set_ylabel(title)
        if len(labels) < 9:
            fig_sub.legend()

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.5)
        fig_sub.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)

    def add_image(self, plot_data, matrix_dimensions, row_index: int, column_index: int, title: str):

        fig_sub = self.fig.add_subplot(self.spec[row_index, column_index])

        matrix = np.zeros(matrix_dimensions)

        for i in range(matrix_dimensions[0]):
            for j in range(matrix_dimensions[1]):
                matrix[i][j] = plot_data[(i, j)][-1]

        im = fig_sub.imshow(matrix, vmin=0, vmax=1) 

        # colorbar
        divider = make_axes_locatable(fig_sub)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(im, cax=cax, orientation='vertical')

        # title and ticks
        fig_sub.set_ylabel(title)
        fig_sub.set_xticks([])
        fig_sub.set_yticks([])

    def generate_plot(self):

        for row in range(self.rows):
            for col in range(self.columns):
        
                graph_index = (row) * self.columns + col

                if graph_index < self.number_of_graphs:

                    print(graph_index)

                    attribute_title = self.plot_keys[graph_index]
                    attribute_config = self.plot_config["data"][attribute_title]
                    attribute_plot_type = attribute_config['plot_type']
                    attribute_key_format = attribute_config['key_format']
                    attribute_scale_axes = attribute_config.get("scale_axes")

                    if attribute_key_format == 'uniform':

                        attribute_keys = attribute_config['keys']
                        attribute_ylimits = attribute_config['ylimits']
                        attribute_labels = attribute_config['labels']

                    elif attribute_key_format == 'recursive':
                        
                        # import pdb; pdb.set_trace()
                        base_attribute_key = attribute_config['keys']
                        fill_ranges = [list(range(r)) for r in attribute_config['key_format_ranges']]
                        fill_combos = list(itertools.product(*fill_ranges))

                        if attribute_plot_type == "scalar":
                            attribute_keys = []
                            attribute_labels = []
                            for fill_combo in fill_combos:
                                attribute_key = copy.deepcopy(base_attribute_key)
                                for i, fill in enumerate(fill_combo):
                                    attribute_key = attribute_key.replace('%', str(fill), 1)
                                attribute_keys.append(attribute_key)
                                attribute_labels.append(tuple(fill_combo))
                        
                        elif attribute_plot_type == 'image':

                            attribute_keys = {}
                            for fill_combo in fill_combos:
                                attribute_key = copy.deepcopy(base_attribute_key)
                                for i, fill in enumerate(fill_combo):
                                    attribute_key = attribute_key.replace('%', str(fill), 1)
                                attribute_keys[tuple(fill_combo)] = attribute_key

                    else:
                        raise ValueError("Key format {} not recognized".format(attribute_key_format))

                    if attribute_plot_type == 'scalar':
                        plot_data = [self.data[attribute_key].dropna().tolist() for attribute_key in attribute_keys]
                        self.add_subplot(
                            plot_data=plot_data, row_index=row, column_index=col, title=attribute_title, 
                            labels=attribute_labels, ylimits=attribute_ylimits, scale_axes=attribute_scale_axes
                            )

                    elif attribute_plot_type == 'image':
                        plot_data = {index: self.data[attribute_keys[index]].dropna().tolist() for index in attribute_keys}
                        self.add_image(
                            plot_data=plot_data, matrix_dimensions=tuple(attribute_config['key_format_ranges']), row_index=row, column_index=col, title=attribute_title
                        )

                    else:
                        raise ValueError("Plot type {} not recognized".format(attribute_plot_type))

        self.fig.suptitle("Summary Plot: {}".format(self.experiment_name))

        self.fig.savefig("{}/summary_plot.pdf".format(self.save_path), dpi=500)
        plt.close()


if __name__ == "__main__":

    os.makedirs("{}/figures/".format(args.save_path), exist_ok=True)
    figure_save_path = "{}/figures".format(args.save_path)

    sP = SummaryPlot(data_csv=args.path_to_csv, plot_config_path=args.summary_plot_attributes, save_path=figure_save_path, experiment_name=args.exp_name)
    sP.generate_plot()



# def summary_plot(data: Dict, path: str, scale_axes: int, start: float, end: float, figure_title: str):
#     """
#     Generate a 2x2 plot with 

#         - generalisation error (log)
#         - teacher-student overlap
#         - student-student overlap
#         - second layer weights

#         or the relevant datasets specified by file at path.
#     """
#     # open json
#     with open(path) as json_file:
#         data_keys = json.load(json_file)

#     figs = []

#     number_of_graphs = len(data_keys.keys())

#     rows = LAYOUTS[number_of_graphs][0]
#     columns = LAYOUTS[number_of_graphs][1]

#     width = 5
#     height = 4

#     heights = [height for _ in range(rows)]
#     widths = [width for _ in range(columns)]

#     for r in range(1):

#         fig = plt.figure(constrained_layout=False, figsize=(columns * width, rows * height))

#         spec = gridspec.GridSpec(nrows=rows, ncols=columns, width_ratios=widths, height_ratios=heights)

#         key_index = 0

#         for row in range(rows):
#             for column in range(columns):

#                 if key_index < number_of_graphs:

#                     fig_sub = fig.add_subplot(spec[row, column])

#                     plot_index = list(data_keys.keys())[key_index]
#                     i_data_keys = data_keys[plot_index]

#                     if i_data_keys['image']:

#                         key_body_prefix, teacher_index, im_dim = i_data_keys['keys'].split('/')
#                         key_body_suffix, x, y = im_dim.split('_')
#                         x, y = int(x), int(y)
#                         xy_tuples = list(itertools.product(list(range(x)), list(range(y))))

#                         all_keys = {(xi, yi): "{}/{}/{}_{}_{}".format(key_body_prefix, teacher_index, key_body_suffix, xi, yi) 
#                                     for (xi, yi) in xy_tuples}

#                         matrix = np.zeros((x, y))

#                         for xi in range(x):
#                             for yi in range(y):
#                                 general_y_data = data[all_keys[(xi, yi)]]['general']['seed_{}'.format(r)]
#                                 matrix[xi][yi] = general_y_data[-1]

#                         im = fig_sub.imshow(matrix, vmin=0, vmax=1) 

#                         # colorbar
#                         divider = make_axes_locatable(fig_sub)
#                         cax = divider.append_axes('right', size='5%', pad=0.05)
#                         fig.colorbar(im, cax=cax, orientation='vertical')

#                         # title and ticks
#                         fig_sub.set_ylabel(i_data_keys["title"])
#                         fig_sub.set_xticks([])
#                         fig_sub.set_yticks([])
                    
#                     else:

#                         for k, key in enumerate(i_data_keys['keys']):

#                             if i_data_keys["general"]:

#                                 general_y_data = data[key]['general']['seed_{}'.format(r)]

#                                 # scale axes
#                                 if scale_axes:
#                                     scaling = scale_axes / len(general_y_data)
#                                     x_data = [i * scaling for i in range(len(general_y_data))]
#                                 else:
#                                     x_data = range(len(general_y_data))
                                
#                                 # crop
#                                 full_dataset_range = len(x_data)
#                                 start_index = int(0.01 * start * full_dataset_range)
#                                 end_index = int(0.01 * end * full_dataset_range)
                                
#                                 # plot
#                                 fig_sub.plot(x_data[start_index: end_index], general_y_data[start_index: end_index], label=i_data_keys['labels'][k])

#                             else:

#                                 t1_y_data = data[key]['teacher1']['seed_{}'.format(r)]
#                                 t2_y_data = data[key]['teacher2']['seed_{}'.format(r)]

#                                 # scale axes
#                                 if scale_axes:
#                                     scaling = scale_axes / len(t1_y_data)
#                                     x_data = [i * scaling for i in range(len(t1_y_data))]
#                                 else:
#                                     x_data = range(len(t1_y_data))
                                
#                                 # crop
#                                 full_dataset_range = len(x_data)
#                                 start_index = int(0.01 * start * full_dataset_range)
#                                 end_index = int(0.01 * end * full_dataset_range)
                                
#                                 # plot
#                                 fig_sub.plot(x_data[start_index: end_index], t1_y_data[start_index: end_index], label='Teacher 1')
#                                 fig_sub.plot(x_data[start_index: end_index], t2_y_data[start_index: end_index], label='Teacher 2')

#                             # labelling
#                             fig_sub.set_xlabel("Step")
#                             fig_sub.set_ylabel(i_data_keys["title"])
#                             # column.set_xticklabels(["{:.3e}".format(t) for t in column.get_xticks()])
#                             fig_sub.legend()

#                             # grids
#                             fig_sub.minorticks_on()
#                             fig_sub.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.5)
#                             fig_sub.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)
                        
#                         key_index += 1
        
#         fig.suptitle("Summary Plot: {}".format(figure_title))

#         figs.append(fig)

#     return figs