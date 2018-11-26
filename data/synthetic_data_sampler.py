import pandas as pd
import numpy as np

from data.preprocessing import preprocess


def get_type_index(columns):
    return np.where(columns == 'type')[0][0]


def get_latlng_indices(columns):
    return [np.where(columns == 'lat')[0][0], np.where(columns == 'lng')[0][0]]


def get_indices_list(columns, name):
    return np.where([name in column.lower() for column in columns])[0]


def extract_category_groups(columns):
    # Create dictionary of category groups by category keywords
    category_columns = [c for c in columns if 'category' in c.lower()]

    keys = set()
    for c in category_columns:
        keys.update(c.split('-'))

    category_dict = {}
    for key in sorted(keys):
        category_dict[key] = [c for c in category_columns if key in c.split('-')]

    del category_dict['category']

    # Get positions in column vector for each category
    category_positions = {}
    for i, c in enumerate(columns):
        if 'category' in c.lower():
            category_positions[c] = i

    # Retrieve category grouping dictionary by category indices in column vector
    category_groups = {}
    for i, c in enumerate(columns):
        if 'category' in c.lower():
            category_position = category_positions[c]
            category_groups[category_position] = set()
            for key, items in category_dict.items():
                if c in items:
                    category_groups[category_position].update([category_positions[category] for category in items])

    return category_groups


def filter_by_type(data, reference_index, type_index, fraction=0.2, verbose=False):
    types = data[:, type_index]

    if verbose:
        print('\tType:', types[reference_index])

    return np.where(types == types[reference_index])[0]


def filter_by_latlng(data, reference_index, latlng_indices, fraction=0.2, verbose=False):
    latlng = data[:, latlng_indices]
    std = np.std(latlng, axis=0)
    latlng_dist = np.abs(latlng - latlng[reference_index])
    filtered_indices = np.where(np.all(latlng_dist < fraction * std, axis=1))[0]

    if verbose:
        print('\tReference latlng:', latlng[reference_index])

    return filtered_indices


def filter_by_category(data, reference_index, category_groups, fraction=0.2, verbose=False):
    category_indices = np.array(list(category_groups))
    reference_categories = category_indices[np.where(data[reference_index, category_indices] > 0)[0]]

    categories = set()
    for category in reference_categories:
        categories |= category_groups[category]
    categories = list(categories)

    if verbose:
        print('\tReference categories:', reference_categories)
        print('\tCategories to filter:', categories)

    selected_cols = data[:, categories]
    sum_selected_cols = np.sum(selected_cols, axis=1)
    filtered_indices = np.where(sum_selected_cols > 0)[0]

    return filtered_indices


def filter_by_utility(data, reference_index, utility_indices, fraction=0.2, verbose=False):
    utilities = data[:, utility_indices]
    reference_utilities = np.where(utilities[reference_index] > 0)[0]

    if verbose:
        print('\tReference utilities:', reference_utilities)

    filtered_indices = np.where(np.sum(utilities[:, reference_utilities], axis=1) > 0)[0]

    return filtered_indices


def filter_by_features(data, reference_index, features_indices, fraction=0.5, verbose=False):
    features = data[:, features_indices]
    reference_features = np.where(features[reference_index] > 0)[0]

    if verbose:
        print('\tReference features:', reference_features)

    if len(reference_features) < 5:
        return np.arange(len(data))

    filtered_indices = \
    np.where(np.sum(features[:, reference_features], axis=1) > int(fraction * len(reference_features)))[0]

    return filtered_indices


def filter_by_price_or_area(data, reference_index, pa_indices, fraction=0.2, verbose=False):
    pas = data[:, pa_indices]
    sum_pas = np.sum(pas, axis=1)
    defined_pas = np.count_nonzero(pas, axis=1)
    rows_with_pas = np.nonzero(defined_pas)[0]

    if verbose:
        print('\tPAs shape:', pas.shape)
        print('\tSum PAs shape:', sum_pas.shape)
        print('\tDefined PAs shape:', defined_pas.shape)
        print('\tRows with PAs shape:', rows_with_pas.shape)
        print()
        print('\tRows with PAs:', len(rows_with_pas))

    reference_pas = data[reference_index, pa_indices]
    reference_pas = reference_pas[np.nonzero(reference_pas)]

    if len(reference_pas) == 0 or len(rows_with_pas) <= 1:
        return np.arange(len(data))

    mean_pas = sum_pas[rows_with_pas] / defined_pas[rows_with_pas]
    std_pa = np.std(mean_pas)

    if verbose:
        print('\tMean PAs shape:', mean_pas.shape)
        print('\tMean PA:', np.mean(mean_pas))
        print('\tPA deviation:', std_pa)

    if std_pa == 0:
        return np.arange(len(data))

    reference_pa = np.mean(reference_pas)

    def _process_row(row):
        row_pas = row[pa_indices]
        row_pas = row_pas[np.nonzero(row_pas)]

        if len(row_pas) == 0:
            return True

        row_pa = np.mean(row_pas)
        return np.abs(reference_pa - row_pa) < fraction * std_pa

    filtered_indices = np.where(np.apply_along_axis(_process_row, axis=1, arr=data))[0]

    return filtered_indices


def get_filters(columns,
                type_filter=True,
                latlng_param=0.2,
                utility_filter=True,
                features_param=0.5,
                category_filter=True,
                price_param=0.6,
                area_param=0.7):
    filters = []

    if type_filter:
        filters.append((get_type_index(columns), filter_by_type, None))

    if latlng_param is not None:
        filters.append((get_latlng_indices(columns), filter_by_latlng, latlng_param))

    if utility_filter:
        filters.append((get_indices_list(columns, 'utility'), filter_by_utility, None))

    if features_param is not None:
        filters.append((get_indices_list(columns, 'feature'), filter_by_features, features_param))

    if category_filter:
        filters.append((extract_category_groups(columns), filter_by_category, None))

    if price_param is not None:
        filters.append((get_indices_list(columns, 'price'), filter_by_price_or_area, price_param))

    if area_param is not None:
        filters.append((get_indices_list(columns, 'value-area'), filter_by_price_or_area, area_param))

    return filters


def get_synthetic_user(data, reference_index, filters, verbose=False):
    if verbose:
        print('All data shape:', data.shape)
        print('Random reference index:', reference_index)

    curr_reference = reference_index
    prev_filtered_indices = np.arange(len(data))
    filtered_indices = None
    for indices, method, param in filters:
        if verbose:
            print()
            print(method.__name__)

        indices_to_filter = method(data[prev_filtered_indices], curr_reference, indices, fraction=param,
                                   verbose=verbose)
        filtered_indices = prev_filtered_indices[indices_to_filter.astype(np.int32)]

        if verbose:
            print("Shape:", filtered_indices.shape)
            print('Filtered', (1 - len(filtered_indices) / len(prev_filtered_indices)) * 100, '%')

        curr_reference = np.where(filtered_indices == reference_index)[0][0]
        assert np.all(data[filtered_indices][curr_reference] == data[reference_index])
        prev_filtered_indices = filtered_indices

    rewards = np.ones(len(data)) * -1
    rewards[filtered_indices] = 1

    return rewards, curr_reference


def retrieve_synthetic_data(data=None,
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
                            min_positive=50,
                            max_positive=100,
                            verbose=False):
    if data is None:
        data = preprocess(pd.read_pickle(input_path))
    np_data = data.values

    fst_filters = get_filters(data.columns,
                              type_filter=fst_type_filter,
                              latlng_param=fst_latlng_param,
                              utility_filter=fst_utility_filter,
                              features_param=fst_feature_param,
                              category_filter=fst_category_filter,
                              price_param=fst_price_param,
                              area_param=fst_area_param)

    snd_filters = get_filters(data.columns,
                              type_filter=snd_type_filter,
                              latlng_param=snd_latlng_param,
                              utility_filter=snd_utility_filter,
                              features_param=snd_feature_param,
                              category_filter=snd_category_filter,
                              price_param=snd_price_param,
                              area_param=snd_area_param)

    counter = 0
    while True:
        counter += 1
        reference_index = np.random.randint(data.shape[0])
        rewards, reference_index = get_synthetic_user(np_data, reference_index, fst_filters, verbose)
        selected = np_data[np.where(rewards > 0)[0]]

        if len(selected) >= max_selected:
            continue

        rewards, _ = get_synthetic_user(selected, reference_index, snd_filters, verbose)
        positive = np.where(rewards > 0)[0]

        if min_positive <= len(positive) <= max_positive:
            break

    return selected, rewards
