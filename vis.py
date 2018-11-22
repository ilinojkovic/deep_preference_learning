import matplotlib.pyplot as plt
import sys

from core.summary_writer import SummaryWriter
from visualization.visualizer import Visualizer


def main():
    if len(sys.argv) < 2:
        raise ValueError('At least one file or folder needs to be provided')
    results = SummaryWriter.loads(*sys.argv[1:])
    vis = Visualizer(results)

    f = plt.figure()

    ax = f.add_subplot(221)
    vis.plot_metric(lambda s: s.precision, ax=ax, print_params=True)
    ax.set_title('Precision')

    ax = f.add_subplot(222)
    vis.plot_metric(lambda s: s.recall, ax=ax)
    ax.set_title('Recall')

    ax = f.add_subplot(223)
    vis.plot_metric(lambda s: s.cost, ax=ax)
    ax.set_title('Cost')

    ax = f.add_subplot(224)
    vis.plot_metric(lambda s: s.tp + s.fn, ax=ax)
    ax.set_title('Sampled Positives')

    plt.show()


if __name__ == '__main__':
    main()
