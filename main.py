from absl import app
from absl import flags
import datetime
from optparse import OptionParser
import pandas as pd

from algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from algorithms.reward_distribution_sampling import RewardDistributionSampling
from core.bandit_dataset import BanditDataset
from data.synthetic_data_sampler import retrieve_synthetic_data, preprocess
from evaluation.grid_search import grid_hparams
from evaluation.summary import Summary

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '../logs/tf/', 'Base directory to save output')
flags.DEFINE_bool('verbose', True, 'Print runtime progress statements')
flags.DEFINE_integer('checkpoint_freq', 100, 'Print checkpoint steps.')
LOG_PATH = '../logs/my/'


def create_parser():
    parser = OptionParser()
    parser.add_option('-i', '--input',
                      dest='input',
                      default='../data/processed/geo_all.pkl',
                      help='Path to .xml file to extract property data from.')
    return parser


def run_configuration(algorithm, hparams, data):
    if FLAGS.verbose:
        print('Running configuration:')
        for param_key, param_value in hparams.values().items():
            print('\t', param_key, '=', param_value)
        print()

    data.positive_reward = hparams.positive_reward
    data.negative_reward = hparams.negative_reward

    sampling = algorithm(hparams, data)
    for step in range(hparams.num_steps):
        if FLAGS.verbose and (step + 1) % FLAGS.checkpoint_freq == 0:
            print('>> Step:', step + 1)
            hparams.show_training = True
        else:
            hparams.show_training = False

        if step >= hparams.positive_start:
            action_i, action, pred_r, opt_r = sampling.action()
        else:
            action_i = step
            action = sampling.data.actions[step]
            pred_r = opt_r = sampling.data.rewards[step]
        sampling.update(action_i, action, pred_r, opt_r)

    return sampling.h_data


def main(_):
    # Load command input options
    options, remainder = create_parser().parse_args()

    num_users_to_run = 200
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
        'lambda_prior': 0.1
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
        'training_freq_network': 1,
        'training_epochs': 40,
        'show_training': True,
        'freq_summary': 20,
        'lr_decay_rate': 0.99,
        'positive_reward': 10,
        'negative_reward': -1,
        'positive_start': 0,
    }
    algos.append((RewardDistributionSampling, hparams_reward_dist_grid))

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
        'training_freq_network': [1, 10],
        'training_epochs': 50,
        'show_training': True,
        'freq_summary': 25,
        'lr_decay_rate': 0.99,
        'positive_reward': 10,
        'negative_reward': -1,
        'positive_start': 0,
        'a0': 6,
        'b0': 6,
        'lambda_prior': 0.1
    }
    algos.append((NeuralLinearPosteriorSampling, hparams_neural_lin_grid))

    raw_data = preprocess(pd.read_pickle(options.input))

    for i in range(num_users_to_run):
        actions, rewards = retrieve_synthetic_data(data=raw_data)

        data = BanditDataset(actions, rewards, hparams_linear_grid['positive_start'])

        if FLAGS.verbose:
            print('==> User {} generated <=='.format(str(i).zfill(2)))
            print('No. selected actions:', data.num_actions)
            print('Positive examples:', len(data.positive_actions))
            print()

        for algo, hparams_grid in algos:
            for hparams in grid_hparams(hparams_grid):
                h_data = run_configuration(algo, hparams, data)
                Summary(hparams, h_data).save(log_path)

        if FLAGS.verbose:
            print('\n================================\n\n')

    return 0


if __name__ == '__main__':
    app.run(main)
