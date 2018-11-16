"""Define class for storing all historical actions, predicted and optimal rewards"""
import numpy as np


class SummaryWriter(object):

    def __init__(self, hparams):
        self.hparams = hparams
        self._actions = []
        self._pred_rewards = []
        self._opt_rewards = []

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
        if not hasattr(self, _name):
            setattr(self, _name, [val])
            setattr(SummaryWriter, name, property(lambda _self: np.array(getattr(_self, _name))))
        else:
            arr = getattr(self, _name)
            if cum:
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

    @property
    def actions(self):
        return np.array(self._actions)

    @property
    def pred_rewards(self):
        return np.array(self._pred_rewards)

    @property
    def opt_rewards(self):
        return np.array(self._opt_rewards)
