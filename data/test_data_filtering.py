from optparse import OptionParser
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.synthetic_data_sampler import preprocess, retrieve_synthetic_data


def create_parser():
    parser = OptionParser()
    parser.add_option('-i', '--input',
                      dest='input',
                      default='../data/processed/geo_all.pkl',
                      help='Path to .xml file to extract property data from.')
    return parser


def main():
    # Load command input options
    options, remainder = create_parser().parse_args()

    data = preprocess(pd.read_pickle(options.input))

    sample_synthetic_sizes = []
    selected_sample_sizes = []
    for i in range(1000):
        if (i + 1) % 100 == 0:
            print(i + 1)

        _, rewards = retrieve_synthetic_data(data=data,
                                             fst_type_filter=True,
                                             fst_latlng_param=0.5,
                                             snd_type_filter=True,
                                             snd_latlng_param=1,
                                             snd_utility_filter=True,
                                             snd_feature_param=0.5,
                                             snd_category_filter=True,
                                             snd_price_param=0.6,
                                             snd_area_param=0.7,
                                             max_selected=float('inf'),
                                             min_positive=-1,
                                             max_positive=float('inf'),
                                             verbose=False)
        selected_sample_sizes.append(len(rewards))
        sample_synthetic_sizes.append(np.count_nonzero(rewards == 1))
    sample_synthetic_sizes = np.array(sample_synthetic_sizes)
    selected_sample_sizes = np.array(selected_sample_sizes)

    print('Positives')
    print('Non-zero', np.count_nonzero(sample_synthetic_sizes))
    print('Mean', np.mean(sample_synthetic_sizes))
    print('Median', np.median(sample_synthetic_sizes))

    print('\nSelected')
    print('Non-zero', np.count_nonzero(selected_sample_sizes))
    print('Mean', np.mean(selected_sample_sizes))
    print('Median', np.median(selected_sample_sizes))

    num_bins = 20
    f = plt.figure()
    ax1 = f.add_subplot(311)
    ax1.hist(sample_synthetic_sizes, num_bins, color='tab:orange')
    ax1.set_title('Positives')

    ax2 = f.add_subplot(312)
    ax2.hist(selected_sample_sizes, num_bins, color='tab:green')
    ax2.set_title('Selected')

    ax3 = f.add_subplot(313)
    ax3.hist(sample_synthetic_sizes / selected_sample_sizes, num_bins, color='tab:red')
    ax3.set_title('Ratio')
    plt.show()


if __name__ == '__main__':
    main()
