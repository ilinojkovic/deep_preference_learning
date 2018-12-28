import sys
import matplotlib.pyplot as plt
import numpy as np

from evaluation.summary import Summary
from visualization.visualizer import Visualizer


def main():
    if len(sys.argv) < 2:
        raise ValueError('At least one file or folder needs to be provided')
    results = Summary.loads(*sys.argv[1:])
    vis = Visualizer(results)

    f = plt.figure()

    ax = f.add_subplot(231)
    vis.plot_metric(lambda s: s.precision, name='Precision', ax=ax, legend=False, print_params=True)
    ax = f.add_subplot(232)
    vis.plot_metric(lambda s: s.tp + s.fn, name='Sampled Positives', ax=ax, legend=False)

    def _f1(p, r):
        f1 = np.zeros(shape=p.shape)
        pr_sum = p + r
        i = pr_sum > 0
        f1[i] = 2 * p[i] * r[i] / pr_sum[i]

        return f1

    ax = f.add_subplot(233)
    vis.plot_metric(lambda s: _f1(s.precision, s.recall), name='F1 score', ax=ax, legend=False)
    ax = f.add_subplot(234)
    vis.plot_metric(lambda s: s.recall, name='Recall', ax=ax, legend=False)
    ax = f.add_subplot(235)
    vis.plot_metric(lambda s: s.unique, name='Number of unique actions', ax=ax, legend=False)
    ax = f.add_subplot(236)
    vis.plot_metric(lambda s: s.unique_positive, name='Number of unique positive actions', ax=ax, legend=True)

    plt.show()


if __name__ == '__main__':
    main()
