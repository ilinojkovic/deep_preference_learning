from absl import app
from absl import flags
import datetime
import numpy as np
from optparse import OptionParser
import pandas as pd
from sklearn.decomposition import PCA

from algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from algorithms.reward_distribution_sampling import RewardDistributionSampling
from core.bandit_dataset import BanditDataset
from data.synthetic_data_sampler import retrieve_synthetic_data
from data.preprocessing import preprocess, normalize
from evaluation.grid_search import grid_hparams
from evaluation.summary import Summary

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '../logs/tf/', 'Base directory to save output')
flags.DEFINE_bool('sampling_verbose', True, 'Print runtime progress statements')
flags.DEFINE_integer('checkpoint_freq', 100, 'Print checkpoint steps.')
LOG_PATH = '../logs/my/'


def create_parser():
    parser = OptionParser()
    parser.add_option('-i', '--input',
                      dest='input',
                      default='../data/processed/geo_all.pkl',
                      help='Path to .xml file to extract property data from.')
    return parser


def main(_):
    # Load command input options
    options, remainder = create_parser().parse_args()

    num_users_to_run = 10
    log_path = LOG_PATH + datetime.datetime.now().strftime('%y%m%d%H%M%S%f')

    algos = []

    hparams_linear_grid = {
        'name': 'LinearFullPosterior',
        'num_steps': 500,
        'actions_dim': 98,
        'init_scale': 0.3,
        'positive_reward': 10,
        'negative_reward': -1,
        'positive_start': 0,
        'a0': 6,
        'b0': 6,
        'lambda_prior': 0.1,
        'pca': [10],
        'remove_actions': False,
        'intercept': True,
    }
    algos.append((LinearFullPosteriorSampling, hparams_linear_grid))

    hparams_reward_dist_grid = {
        'name': 'RewardDistribution',
        'num_steps': 500,
        'actions_dim': 98,
        'init_scale': 0.3,
        'activation': 'relu',
        'layer_sizes': [[50, 50]],
        'batch_size': 32,
        'reset_lr': True,
        'activate_decay': True,
        'initial_lr': 0.1,
        'max_grad_norm': 5.0,
        'training_freq_network': [1, 10],
        'training_epochs': 40,
        'show_training': True,
        'freq_summary': 20,
        'lr_decay_rate': 0.99,
        'positive_reward': 10,
        'negative_reward': -1,
        'positive_start': 0,
        'pca': [True, False],
        'remove_actions': True,
    }
    # algos.append((RewardDistributionSampling, hparams_reward_dist_grid))

    hparams_neural_lin_grid = {
        'name': 'NeuralLinear',
        'num_steps': 500,
        'actions_dim': 98,
        'init_scale': 0.3,
        'activation': 'relu',
        'layer_sizes': [[50, 50]],
        'batch_size': 32,
        'reset_lr': True,
        'activate_decay': True,
        'initial_lr': 0.1,
        'max_grad_norm': 5.0,
        'training_freq': 1,
        'training_freq_network': [10],
        'training_epochs': 50,
        'show_training': True,
        'freq_summary': 25,
        'lr_decay_rate': 0.99,
        'positive_reward': 10,
        'negative_reward': -1,
        'positive_start': 0,
        'a0': 6,
        'b0': 6,
        'lambda_prior': 0.1,
        'pca': [10],
        'remove_actions': False,
    }
    algos.append((NeuralLinearPosteriorSampling, hparams_neural_lin_grid))

    raw_data = pd.read_pickle(options.input)
    raw_data = preprocess(raw_data)

    for i in range(num_users_to_run):
        actions, rewards = retrieve_synthetic_data(data=raw_data,
                                                   fst_type_filter=True,
                                                   fst_geo_param=0.7,
                                                   fst_category_filter=None,
                                                   fst_price_param=None,
                                                   fst_area_param=None,
                                                   fst_feature_param=None,
                                                   snd_type_filter=True,
                                                   snd_geo_param=1,
                                                   snd_category_filter=True,
                                                   snd_price_param=0.08,
                                                   snd_area_param=0.1,
                                                   snd_feature_param=0.5,
                                                   max_selected=6000,
                                                   min_positive=20,
                                                   max_positive=1000,
                                                   verbose=False)

        normalized_actions = normalize(actions)
        data = BanditDataset(normalized_actions, rewards)

        if FLAGS.sampling_verbose:
            print('==> User {} generated <=='.format(str(i).zfill(2)))
            print('Actions shape:', data.actions.shape)
            print('Positive examples:', len(data.positive_actions))
            print()

        for algo, hparams_grid in algos:
            for hparams in grid_hparams(hparams_grid):
                if hparams.pca:
                    pca = PCA(hparams.pca, whiten=True)
                    pca_actions = pca.fit_transform(data.actions)

                    run_data = BanditDataset(pca_actions, rewards)
                else:
                    run_data = data

                hparams.set_hparam('actions_dim', run_data.actions_dim)

                sampling = algo(hparams, run_data)
                summary = sampling.run()
                summary.save(log_path)

        if FLAGS.sampling_verbose:
            print('\n================================\n\n')

    return 0


if __name__ == '__main__':
    app.run(main)
