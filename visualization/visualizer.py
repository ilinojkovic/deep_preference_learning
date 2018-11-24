import matplotlib.pyplot as plt

from core.utils import plot_mean_and_ci, get_mean_and_ci


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

    def plot_metric(self, metric, ax=None, name=None, print_params=False):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(111)

        model_lines = []
        model_names = []
        self.curr_color = 0
        for model_data in self.data:
            if len(model_data) == 0:
                continue
            mean, lcb, ucb = get_mean_and_ci(metric, model_data)
            line = plot_mean_and_ci(mean, lcb, ucb, ax,
                                    color_mean=self.colors[self.curr_color],
                                    color_shading=self.colors[self.curr_color])
            self.curr_color = (self.curr_color + 1) % len(self.colors)
            model_lines.append(line)
            model_names.append(model_data[0].hparams.name[:3] + '_' + model_data[0].hparams.id)

            if print_params:
                for param_key, param_value in model_data[0].hparams.values().items():
                    print('\t', param_key, '=', param_value)
                print()

        ax.legend(model_lines, model_names, loc='center left', bbox_to_anchor=(1, 0.5))

        if name is not None:
            ax.set_title(name)

        if ax is None:
            plt.show(block=False)
