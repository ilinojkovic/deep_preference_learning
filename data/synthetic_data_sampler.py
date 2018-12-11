import pandas as pd
import numpy as np

from data.preprocessing import get_indices_list


def get_latlng_indices(columns):
    return [np.where(columns == 'lat')[0][0], np.where(columns == 'lng')[0][0]]


def filter_categorical(data, reference_index, column_indices):
    features = data[:, column_indices]
    return np.where(np.all(features == features[reference_index], axis=1))[0]


def filter_numerical(data, reference_index, column_indices, fraction_of_std=1):
    features = data[:, column_indices]
    masked_features = np.ma.array(features, mask=(features == 0))
    std = np.std(masked_features, axis=0).data
    filtered = np.where(np.all(np.abs(features - features[reference_index]) < fraction_of_std * std, axis=1))[0]
    missing_data = np.where(np.all(features == 0, axis=1))[0]
    return np.concatenate((filtered, missing_data))


def filter_by_features(data, reference_index, features_indices, fraction=0.5, verbose=False):
    features = data[:, features_indices]
    reference_features = np.where(features[reference_index] > 0)[0]

    if verbose:
        print('\tReference features:', reference_features)

    if len(reference_features) < 3:
        return np.arange(len(data))

    filtered_indices = \
        np.where(np.sum(features[:, reference_features], axis=1) > int(fraction * len(reference_features)))[0]

    return filtered_indices


def synthetic_user_rewards(data,
                           reference_index,
                           type_filter=True,
                           geo_param=0.2,
                           category_filter=True,
                           price_param=0.08,
                           area_param=0.1,
                           feature_param=0.5,
                           verbose=False):
    num_filters = 0
    rewards = np.zeros(len(data))

    if type_filter:
        num_filters += 1
        filtered = filter_categorical(data.values, reference_index, get_indices_list(data.columns, 'type'))
        rewards[filtered] += 1
        if verbose:
            print('Type:', len(filtered))

    if geo_param:
        num_filters += 1
        filtered = filter_numerical(data.values, reference_index, get_indices_list(data.columns, 'geo'), geo_param)
        rewards[filtered] += 1
        if verbose:
            print('Geo:', len(filtered))

    if category_filter:
        num_filters += 1
        filtered = filter_categorical(data.values, reference_index, get_indices_list(data.columns, 'category'))
        rewards[filtered] += 1
        if verbose:
            print('Category:', len(filtered))

    if price_param:
        num_filters += 1
        filtered = filter_numerical(data.values, reference_index, get_indices_list(data.columns, 'price'), price_param)
        rewards[filtered] += 1
        if verbose:
            print('Price:', len(filtered))

    if area_param:
        num_filters += 1
        filtered = filter_numerical(data.values, reference_index, get_indices_list(data.columns, 'area'), area_param)
        rewards[filtered] += 1
        if verbose:
            print('Area:', len(filtered))

    if feature_param:
        num_filters += 1
        filtered = filter_by_features(data.values, reference_index, get_indices_list(data.columns, 'feature'),
                                      feature_param)
        rewards[filtered] += 1
        if verbose:
            print('Feature:', len(filtered))

    rewards[rewards != num_filters] = 0
    rewards[rewards == num_filters] = 1

    return rewards


def retrieve_synthetic_data(data=None,
                            fst_type_filter=True,
                            fst_geo_param=0.5,
                            fst_category_filter=None,
                            fst_price_param=None,
                            fst_area_param=None,
                            fst_feature_param=None,
                            snd_type_filter=True,
                            snd_geo_param=1,
                            snd_category_filter=True,
                            snd_price_param=0.6,
                            snd_area_param=0.7,
                            snd_feature_param=0.5,
                            max_selected=6000,
                            min_positive=20,
                            max_positive=200,
                            verbose=False):
    counter = 0
    while True:
        counter += 1
        reference_index = np.random.randint(data.shape[0])
        selected_rewards = synthetic_user_rewards(data,
                                                  reference_index,
                                                  type_filter=fst_type_filter,
                                                  geo_param=fst_geo_param,
                                                  category_filter=fst_category_filter,
                                                  price_param=fst_price_param,
                                                  area_param=fst_area_param,
                                                  feature_param=fst_feature_param,
                                                  verbose=verbose)
        selected = data.iloc[np.where(selected_rewards > 0)[0]]

        if len(selected) >= max_selected:
            continue

        rewards = synthetic_user_rewards(data,
                                         reference_index,
                                         type_filter=snd_type_filter,
                                         geo_param=snd_geo_param,
                                         category_filter=snd_category_filter,
                                         price_param=snd_price_param,
                                         area_param=snd_area_param,
                                         feature_param=snd_feature_param,
                                         verbose=verbose)
        positive = np.count_nonzero(rewards)
        if min_positive <= len(positive) <= max_positive:
            break

    return selected, rewards
