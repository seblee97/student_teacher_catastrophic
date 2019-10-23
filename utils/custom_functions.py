import numpy as np
import matplotlib.pyplot as plt

def visualise_matrix(matrix_data: np.ndarray, fig_title: str=None):
    """
    Show heatmap of matrix

    :param matrix data: (M x N) numpy array containing matrix data
    :param fig_title: title to be given to figure
    :return fig: matplotlib figure with visualisation of matrix data
    """
    fig = plt.figure()
    plt.imshow(matrix_data)
    if fig_title:
        fig.suptitle(fig_title, fontsize=20)
    plt.close()
    return fig
