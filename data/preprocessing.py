import numpy as np
import pandas as pd


def preprocess(df, errors='coerce'):
    data = []
    for key in df:
        nd_column = pd.to_numeric(df[key], errors=errors)
        if nd_column.count() > 0:
            data.append(nd_column)
    preprocessed_df = pd.concat(data, axis=1, keys=[s.name for s in data])
    preprocessed_df = preprocessed_df.fillna(0)

    col_reg = ['attach', 'price', 'category', 'feature', 'value', 'utility', 'lat', 'lng', 'type']
    cols = [c for c in preprocessed_df.columns if any([reg in c.lower() for reg in col_reg])]
    preprocessed_df = preprocessed_df[cols]

    return preprocessed_df


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


def normalize(data):
    std = np.std(data, axis=0)
    to_divide = np.where(std != 0)[0]

    normalized = data - np.mean(data, axis=0)
    normalized[:, to_divide] /= std[to_divide]

    return normalized
