import datetime
from optparse import OptionParser
import pandas as pd

from algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from algorithms.reward_distribution_sampling import RewardDistributionSampling
from core.bandit_dataset import BanditDataset
from core.summary_writer import SummaryWriter
from data.synthetic_data_sampler import retrieve_synthetic_data, preprocess
from evaluation.grid_search import grid_hparams
from evaluation.run import run_configuration


LOG_PATH = '../logs/'


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

    num_users_to_run = 5
    log_path = LOG_PATH + datetime.datetime.now().strftime('%y%m%d%H%M%S%f')

    algos = []

    hparams_linear_grid = {
        'name': 'LinearFullPosterior',
        'num_steps': 600,
        'actions_dim': 98,
        'init_scale': 0.3,
        'show_training': True,
        'freq_summary': 200,
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
        'activation': 'relu',
        'layer_sizes': [False],
        'batch_size': 1,
        'activate_decay': True,
        'initial_lr': 0.1,
        'max_grad_norm': 5.0,
        'show_training': True,
        'freq_summary': 200,
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

    for i in range(num_users_to_run):
        actions, rewards = retrieve_synthetic_data(data=raw_data)

        data = BanditDataset(actions, rewards, hparams_linear_grid['positive_start'])

        print('==> User {} generated <=='.format(str(i).zfill(2)))
        print('No. selected actions:', data.num_actions)
        print('Positive examples:', len(data.positive_actions))
        print()

        for algo, hparams_grid in algos:
            for hparams in grid_hparams(hparams_grid):
                summary = run_configuration(algo, hparams, data)
                SummaryWriter.save(summary, log_path)


if __name__ == '__main__':
    main()
