"""Define class for storing all historical actions, predicted and optimal rewards"""
import datetime
import errno
import numpy as np
import os
import pickle


class SummaryWriter(object):

    def __init__(self, hparams):
        self.hparams = hparams
        self._actions = []
        self._pred_rewards = []
        self._opt_rewards = []

    def __getattr__(self, item):
        _item = '_' + item
        if _item in self.__dict__:
            return np.array(getattr(self, _item))
        else:
            setattr(self, item, [])
            return getattr(self, item)

    def add(self, action_i, pred_r, opt_r):
        self._actions.append(action_i)
        self._pred_rewards.append(pred_r)
        self._opt_rewards.append(opt_r)

        self._add_cost()
        self._add_tp()
        self._add_fp()
        self._add_tn()
        self._add_fn()
        self._add_precision()
        self._add_recall()

    def _add_attr(self, name, val, cum=False):
        _name = '_' + name
        arr = getattr(self, _name)
        if len(arr) > 0 and cum:
            val += arr[-1]
        arr.append(val)

    def _add_cost(self):
        cost = abs(self.pred_rewards[-1] - self.opt_rewards[-1])
        self._add_attr('cost', cost)

    def _add_tp(self):
        tp = int(self.opt_rewards[-1] > 0 and self.pred_rewards[-1] > self.hparams.positive_reward / 2)
        self._add_attr('tp', tp, cum=True)

    def _add_fp(self):
        fp = int(self.opt_rewards[-1] <= 0 and self.pred_rewards[-1] > self.hparams.positive_reward / 2)
        self._add_attr('fp', fp, cum=True)

    def _add_tn(self):
        tn = int(self.opt_rewards[-1] <= 0 and self.pred_rewards[-1] <= self.hparams.positive_reward / 2)
        self._add_attr('tn', tn, cum=True)

    def _add_fn(self):
        fn = int(self.opt_rewards[-1] > 0 and self.pred_rewards[-1] <= self.hparams.positive_reward / 2)
        self._add_attr('fn', fn, cum=True)

    def _add_precision(self):
        if (self.tp[-1] + self.fp[-1]) == 0:
            precision = 0
        else:
            precision = self.tp[-1] / (self.tp[-1] + self.fp[-1])
        self._add_attr('precision', precision)

    def _add_recall(self):
        if (self.tp[-1] + self.fn[-1]) == 0:
            recall = 0
        else:
            recall = self.tp[-1] / (self.tp[-1] + self.fn[-1])
        self._add_attr('recall', recall)

    @staticmethod
    def save(instance, path):
        model_dir = os.path.join(path, instance.hparams.id)
        instance_path = os.path.join(model_dir, datetime.datetime.now().strftime('%y%m%d%H%M%S%f') + '.pkl')
        try:
            os.makedirs(model_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
        with open(instance_path, 'wb') as f:
            # Hack
            pickle.dump(instance.__dict__, f)

    @staticmethod
    def load(path):
        """Recursively loads SummaryWriter files from directory"""
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                # Unhack
                s = SummaryWriter(None)
                s.__dict__ = pickle.load(f)
                return s

        directory = os.fsencode(path)
        summaries = []
        for fod in os.listdir(directory):
            fod_name = os.fsdecode(fod)
            fod_path = os.path.join(path, fod_name)
            summaries.append(SummaryWriter.load(fod_path))
        return summaries

    @staticmethod
    def loads(*paths):
        """Iteratively calls SummaryWriter.load for each path from the argument list"""
        summaries = []
        for path in paths:
            summaries.extend(SummaryWriter.load(path))
        return summaries
