import numpy as np
import pandas as pd


def to_numeric(df, errors='coerce'):
    data = []
    for key in df:
        nd_column = pd.to_numeric(df[key], errors=errors)
        if nd_column.count() > 0:
            data.append(nd_column)
    preprocessed_df = pd.concat(data, axis=1, keys=[s.name for s in data])
    preprocessed_df = preprocessed_df.fillna(0)

    return preprocessed_df


def filter_columns(df):
    col_reg = ['attach', 'price', 'category', 'feature', 'value', 'lat', 'lng', 'type']
    cols = [c for c in df.columns if any([reg in c.lower() for reg in col_reg])]
    cols.remove('zoneTypes')
    return df[cols]


def remove_outlier_vals(df, m=10):
    data = df.values
    val_ind = np.where(['value' in column.lower() for column in df.columns])[0]

    vals = data[:, val_ind]
    mvals = np.ma.masked_array(vals, mask=(vals == 0))

    medians = np.ma.median(mvals, axis=0)
    d = np.abs(mvals - medians)
    mdev = np.ma.median(d, axis=0)
    s = np.zeros(d.shape)
    s[:, mdev > 0] = d[:, mdev > 0] / mdev

    vals[s > m] = 0
    data[:, val_ind] = vals

    return pd.DataFrame(data=data, index=range(data.shape[0]), columns=df.columns)


def get_indices_list(columns, name):
    return np.where([name in column.lower() for column in columns])[0]


def recategorize(df):
    recategorization_dict = {
        'category-flat': np.array(['category-attic-flat', 'category-flat', 'category-furnished-flat',
                                   'category-loft', 'category-maisonette', 'category-multiple-dwelling',
                                   'category-roof-flat', 'category-studio', 'category-terrace-flat',
                                   'category-hobby-room', 'category-single-room']),
        'category-house': np.array(['category-bifamiliar-house', 'category-castle', 'category-chalet',
                                    'category-farm', 'category-farm-house', 'category-house',
                                    'category-maisonette', 'category-multiple-dwelling',
                                    'category-row-house', 'category-rustico', 'category-single-house',
                                    'category-terrace-house', 'category-villa']),
        'category-land': np.array(['category-building-land', 'category-plot']),
        'category-business': np.array(['category-cafe-bar', 'category-factory', 'category-hotel',
                                       'category-office', 'category-residential-commercial-building',
                                       'category-retail', 'category-retail-space', 'category-warehouse',
                                       'category-workshop']),
        'category-parking': np.array(['category-covered-slot', 'category-garage', 'category-open-slot',
                                      'category-parking-space', 'category-underground-slot'])
    }
    curr_cat_indices = get_indices_list(df.columns, 'category')
    cat_to_ind = {}
    for i in curr_cat_indices:
        cat_to_ind[df.columns.values[i]] = i

    recategorization_indices = []
    new_categories = []
    for k in recategorization_dict:
        new_categories.append(k)
        recategorization_indices.append([cat_to_ind[cat] for cat in recategorization_dict[k]])

    new_cat = np.zeros(shape=(len(df), len(new_categories)))
    for i, cats in enumerate(recategorization_indices):
        new_cat[:, i] = (np.sum(df.values[:, cats], axis=1) > 0).astype(np.int32)

    new_columns = np.concatenate((new_categories, np.delete(df.columns.values, curr_cat_indices)))
    new_data = np.concatenate((new_cat, np.delete(df.values, curr_cat_indices, axis=1)), axis=1)

    return pd.DataFrame(data=new_data, index=np.arange(len(new_data)), columns=new_columns)


def average_out_columns(df, column_substring):
    """Merge multiple columns and average out non-zero values"""
    indices = get_indices_list(df.columns, column_substring)
    vals = df.values[:, indices]
    masked_vals = np.ma.array(vals, mask=(vals == 0))
    mean_vals = np.mean(masked_vals, axis=1).data
    ndata = df.drop(labels=df.columns[indices], axis=1)
    ndata[column_substring] = pd.Series(mean_vals, index=ndata.index)
    return ndata


def prune_sparse(df, min_non_zero=10):
    """Remove columns with less then min_non_zero percent of non-zero values"""
    category_indices = get_indices_list(df.columns, 'category')
    non_zero_perc = np.count_nonzero(df.values, axis=0) / len(df.values) * 100
    to_remove = np.where(non_zero_perc < min_non_zero)[0]
    to_remove = np.delete(to_remove, [i for i in range(len(to_remove)) if to_remove[i] in category_indices])

    ndata = df.drop(labels=df.columns.values[to_remove], axis=1)
    return ndata


def preprocess(df, errors='coerce'):
    df = to_numeric(df, errors=errors)
    df = filter_columns(df)
    df = remove_outlier_vals(df)
    df = recategorize(df)
    df = average_out_columns(df, 'price')
    df = average_out_columns(df, 'area')
    df = prune_sparse(df)
    df = df.rename(index=str, columns={'lat': 'geo_lat', 'lng': 'geo_lng'})
    df = df.reindex(sorted(df.columns), axis=1)

    return df
