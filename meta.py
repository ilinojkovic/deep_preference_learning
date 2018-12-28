from absl import app
from absl import flags
import datetime
from optparse import OptionParser
import os
import pandas as pd
import pickle
import tensorflow as tf
import time

from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from core.bandit_dataset import BanditDataset
from data.synthetic_data_sampler import retrieve_synthetic_data
from data.preprocessing import preprocess, remove_outlier_vals, normalize

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '../logs/tf/', 'Base directory to save output')
flags.DEFINE_bool('meta_verbose', True, 'Print runtime progress statements')
flags.DEFINE_bool('sampling_verbose', False, 'Print runtime progress statements')
flags.DEFINE_integer('checkpoint_freq', 20, 'Print checkpoint steps.')
LOG_PATH = '../logs/best/'


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

    num_meta_datasets = 20
    num_generations = 5
    population_size = 30
    num_children = 5
    best_sample = 8
    lucky_few = 4
    chance_of_mutation = 5
    fraction_to_mutate = 5

    hparams_linear = tf.contrib.training.HParams(name='LinearFullPosterior',
                                                 num_steps=20,
                                                 actions_dim=98,
                                                 positive_reward=10,
                                                 negative_reward=-1,
                                                 a0=6.0,
                                                 b0=6.0,
                                                 lambda_prior=0.1,
                                                 intercept=True,
                                                 remove_actions=False)

    raw_data = pd.read_pickle(options.input)
    raw_data = preprocess(raw_data)
    raw_data = remove_outlier_vals(raw_data)

    meta_data = []
    for _ in range(num_meta_datasets):
        actions, rewards = retrieve_synthetic_data(data=raw_data,
                                                   input_path=None,
                                                   fst_type_filter=True,
                                                   fst_latlng_param=0.5,
                                                   fst_utility_filter=None,
                                                   fst_feature_param=None,
                                                   fst_category_filter=None,
                                                   fst_price_param=None,
                                                   fst_area_param=None,
                                                   snd_type_filter=True,
                                                   snd_latlng_param=1,
                                                   snd_utility_filter=True,
                                                   snd_feature_param=0.5,
                                                   snd_category_filter=True,
                                                   snd_price_param=0.6,
                                                   snd_area_param=0.7,
                                                   max_selected=6000,
                                                   min_positive=10,
                                                   max_positive=1000,
                                                   verbose=False)

        normalized_actions = normalize(actions)
        data = BanditDataset(normalized_actions, rewards)
        meta_data.append(data)

    start_time = time.time()
    ga = GeneticAlgorithm(sampling=LinearFullPosteriorSampling,
                          hparams=hparams_linear,
                          meta_data=meta_data,
                          population_size=population_size,
                          num_children=num_children,
                          best_sample=best_sample,
                          lucky_few=lucky_few,
                          chance_of_mutation=chance_of_mutation,
                          fraction_to_mutate=fraction_to_mutate)
    ga.run(num_generations=num_generations)

    best = ga.get_best_individual_from_population()
    print('Finished. Best fitness score:', best[0])
    print('Elapsed time: ', datetime.timedelta(seconds=(time.time() - start_time)))

    best_path = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%y%m%d%H%M%S%f') + '.pkl')
    with open(best_path, 'wb') as f:
        pickle.dump(best, f)

    return 0


if __name__ == '__main__':
    app.run(main)
