"""Define class for storing all historical actions, predicted and optimal rewards"""
import datetime
import errno
import numpy as np
import os
import pickle


class Summary(object):

    def __init__(self, hparams, h_data):
        self.hparams = hparams

        self.cost = np.abs(h_data.pred_rewards - h_data.opt_rewards)

        self.tp = np.cumsum(np.logical_and(
            h_data.opt_rewards > 0,
            h_data.pred_rewards > hparams.positive_reward / 2
        ).astype(np.int32))
        self.fp = np.cumsum(np.logical_and(
            h_data.opt_rewards <= 0,
            h_data.pred_rewards > hparams.positive_reward / 2
        ).astype(np.int32))
        self.tn = np.cumsum(np.logical_and(
            h_data.opt_rewards <= 0,
            h_data.pred_rewards <= hparams.positive_reward / 2
        ).astype(np.int32))
        self.fn = np.cumsum(np.logical_and(
            h_data.opt_rewards > 0,
            h_data.pred_rewards <= hparams.positive_reward / 2
        ).astype(np.int32))

        self.precision = np.zeros(h_data.n)
        ind = (self.tp + self.fp) > 0
        self.precision[ind] = self.tp[ind] / (self.tp[ind] + self.fp[ind])

        self.recall = np.zeros(h_data.n)
        ind = (self.tp + self.fn) > 0
        self.recall[ind] = self.tp[ind] / (self.tp[ind] + self.fn[ind])

        self.unique = np.array([len(np.unique(h_data.action_indices[:i + 1])) for i in range(h_data.n)])

    def save(self, path):
        model_path = os.path.join(path, self.hparams.name[:3] + '_' + self.hparams.id)
        summary_path = os.path.join(model_path, datetime.datetime.now().strftime('%y%m%d%H%M%S%f') + '.pkl')
        try:
            os.makedirs(model_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
        with open(summary_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Recursively loads SummaryWriter files from directory"""
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        directory = os.fsencode(path)
        summaries = []
        for fod in os.listdir(directory):
            fod_name = os.fsdecode(fod)
            fod_path = os.path.join(path, fod_name)
            summaries.append(Summary.load(fod_path))
        return summaries

    @staticmethod
    def loads(*paths):
        """Iteratively calls SummaryWriter.load for each path from the argument list"""
        summaries = []
        for path in paths:
            loaded = Summary.load(path)
            if not isinstance(loaded, list):
                loaded = [[loaded]]
            summaries.extend(loaded)
        return summaries
