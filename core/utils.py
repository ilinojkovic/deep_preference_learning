import numpy as np
import scipy.stats as st


def plot_mean_and_ci(mean, lb, ub, ax, color_mean=None, color_shading=None):
    """Plot mean and confidence interval bounds"""
    # plot the shaded range of the confidence intervals
    ax.fill_between(range(mean.shape[0]), ub, lb,
                    color=color_shading, alpha=.5)
    # plot the mean on top
    line, = ax.plot(mean, color_mean)

    return line


def get_mean_and_ci(metric, data_list):
    """Return mean and 95% confidence interval for certain metric of the list of HistoricalData objects"""
    stacked = np.stack([metric(h_data) for h_data in data_list])
    mean = np.mean(stacked, axis=0)
    sem = st.sem(stacked, axis=0)
    lcb, ucb = st.t.interval(0.95, np.ones(stacked.shape[1]) * (stacked.shape[0] - 1), loc=mean, scale=sem)
    return mean, lcb, ucb
