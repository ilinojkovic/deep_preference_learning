import sys
import matplotlib.pyplot as plt

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
    ax = f.add_subplot(233)
    vis.plot_metric(lambda s: s.cost, name='Cost', ax=ax, legend=False)
    ax = f.add_subplot(234)
    vis.plot_metric(lambda s: s.recall, name='Recall', ax=ax, legend=False)
    ax = f.add_subplot(235)
    vis.plot_metric(lambda s: s.unique, name='Number of unique actions', ax=ax, legend=False)
    ax = f.add_subplot(236)
    vis.plot_metric(lambda s: s.unique_positive, name='Number of unique positive actions', ax=ax, legend=True)

    plt.show()


if __name__ == '__main__':
    main()
