import sys

from evaluation.summary import Summary
from visualization.visualizer import Visualizer


def main():
    if len(sys.argv) < 2:
        raise ValueError('At least one file or folder needs to be provided')
    results = Summary.loads(*sys.argv[1:])
    vis = Visualizer(results)

    vis.plot_metric(lambda s: s.precision, name='Precision', print_params=True)
    vis.plot_metric(lambda s: s.recall, name='Recall')
    vis.plot_metric(lambda s: s.cost, name='Cost')
    vis.plot_metric(lambda s: s.tp + s.fn, name='Sampled Positives')
    vis.plot_metric(lambda s: s.unique, name='Number of unique actions')


if __name__ == '__main__':
    main()
