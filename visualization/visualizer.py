import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from visualization.plot_utils import plot_mean_and_std


class Visualizer(object):
    """Wrapped methods for performance visualization"""

    def __init__(self, data):
        """Get the data to be visualized.

        Args:
            data: List of lists of HistoricalData objects. First level lists are
                  different models to be plotted alongside each other. Second
                  level lists are the results of the same model bootstrapped
                  for different users.
        """
        self.data = data
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        self.curr_color = 0

    @staticmethod
    def _bootstrap(trait, data_list):
        """Return combined information about the list of HistoricalData objects for certain trait"""
        stacked = np.stack([trait(h_data) for h_data in data_list])
        mean = np.mean(stacked, axis=0)
        sem = st.sem(stacked, axis=0)
        lcb, ucb = st.t.interval(0.95, np.ones(stacked.shape[1]) * (stacked.shape[0] - 1), loc=mean, scale=sem)
        return mean, lcb, ucb

    def plot_trait(self, trait, ax=None):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(111)

        model_lines = []
        model_names = []
        for model_data in self.data:
            if len(model_data) == 0:
                continue
            mean, lcb, ucb = Visualizer._bootstrap(trait, model_data)
            line = plot_mean_and_std(mean, lcb, ucb, ax,
                                     color_mean=self.colors[self.curr_color],
                                     color_shading=self.colors[self.curr_color])
            self.curr_color = (self.curr_color + 1) % len(self.colors)
            model_lines.append(line)
            model_names.append(model_data[0].hparams.name)

        ax.legend(model_lines, model_names)

        if ax is None:
            plt.show()
