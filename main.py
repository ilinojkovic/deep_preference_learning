import matplotlib.pyplot as plt
from optparse import OptionParser
import pandas as pd
import tensorflow as tf

from algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from algorithms.reward_distribution_sampling import RewardDistributionSampling
from core.bandit_dataset import BanditDataset
from data.synthetic_data_sampler import retrieve_synthetic_data, preprocess
from evaluation.grid_search import grid_hparams
from evaluation.run import run_configuration
from visualization.visualizer import Visualizer


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

    num_users_to_run = 20

    algos = []

    hparams_linear_grid = {
        'name': 'LinearFullPosterior',
        'num_steps': 600,
        'actions_dim': 98,
        'init_scale': 0.3,
        'show_training': True,
        'freq_summary': 100,
        'plot_distribution': True,
        'early_stopping': False,
        'patience': 200,
        'positive_reward': 10,
        'negative_reward': -1,
        'positive_start': 0,
        'a0': 6,
        'b0': 6,
        'lambda_prior': 0.25
    }
    algos.append((LinearFullPosteriorSampling, hparams_linear_grid))

    hparams_reward_dist_grid = {
        'name': 'RewardDistribution',
        'num_steps': 600,
        'actions_dim': 98,
        'init_scale': 0.3,
        'activation': tf.nn.relu,
        'layer_sizes': [False],
        'batch_size': 1,
        'activate_decay': True,
        'initial_lr': 0.1,
        'max_grad_norm': 5.0,
        'show_training': True,
        'freq_summary': 100,
        'lr_decay_rate': 0.99,
        'plot_distribution': True,
        'early_stopping': False,
        'patience': 200,
        'positive_reward': 10,
        'negative_reward': -1,
        'positive_start': 0,
    }
    algos.append((RewardDistributionSampling, hparams_reward_dist_grid))

    raw_data = preprocess(pd.read_pickle(options.input))

    results = []
    for _ in range(num_users_to_run):
        actions, rewards = retrieve_synthetic_data(data=raw_data)

        data = BanditDataset(actions, rewards, hparams_linear_grid['positive_start'])

        print('User generated')
        print('No. selected actions:', data.num_actions)
        print('Positive examples:', len(data.positive_actions))
        print()

        model_index = 0
        for algo, hparams_grid in algos:
            for hparams in grid_hparams(hparams_grid):
                h_data = run_configuration(algo, hparams, data)

                if model_index == len(results):
                    results.append([h_data])
                else:
                    results[model_index].append(h_data)
                model_index += 1

    vis = Visualizer(results)

    f = plt.figure()

    ax = f.add_subplot(221)
    vis.plot_trait(lambda h: h.precision, ax=ax)
    ax.set_title('Precision')

    ax = f.add_subplot(222)
    vis.plot_trait(lambda h: h.recall, ax=ax)
    ax.set_title('Recall')

    ax = f.add_subplot(223)
    vis.plot_trait(lambda h: h.cost, ax=ax)
    ax.set_title('Cost')

    ax = f.add_subplot(224)
    vis.plot_trait(lambda h: h.tp + h.fn, ax=ax)
    ax.set_title('Sampled Positives')

    plt.show()


if __name__ == '__main__':
    main()
