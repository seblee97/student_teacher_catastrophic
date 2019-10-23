import numpy as np
import matplotlib.pyplot as plt

def visualise_matrix(matrix_data: np.ndarray, fig_title: str=None, normalised: bool=True):
    """
    Show heatmap of matrix

    :param matrix data: (M x N) numpy array containing matrix data
    :param fig_title: title to be given to figure
    :param normalised: whether or not matrix data is normalised
    :return fig: matplotlib figure with visualisation of matrix data
    """
    fig = plt.figure()
    if normalised:
        plt.imshow(matrix_data, vmin=0, vmax=1)
    else:
        plt.imshow(matrix_data)
    plt.colorbar()
    if fig_title:
        fig.suptitle(fig_title, fontsize=20)
    plt.close()
    return fig
