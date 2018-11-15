import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np


def plot_mean_and_std(mean, lb, ub, ax, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    ax.fill_between(range(mean.shape[0]), ub, lb,
                    color=color_shading, alpha=.5)
    # plot the mean on top
    line, = ax.plot(mean, color_mean)

    return line


def get_faded_color(color, bg=np.array([1, 1, 1])):
    # with alpha = .5, the faded color is the average of the background and color
    return (np.array(cc.to_rgb(color)) + bg) / 2.0


class LegendObject(object):
    def __init__(self, face_color='red', edge_color='white', dashed=False):
        self.face_color = face_color
        self.edge_color = edge_color
        self.dashed = dashed

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.face_color,
            # and whose edges are the faded color
            edgecolor=self.edge_color, lw=3)
        handlebox.add_artist(patch)

        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2 * width / 5, y0], width / 5, height, facecolor=self.edge_color,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)

        return patch
