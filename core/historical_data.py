"""Define class for storing all historical actions, predicted and optimal rewards"""
import numpy as np


class HistoricalData(object):

    def __init__(self, hparams):
        self.hparams = hparams
        self._actions = np.empty(hparams.num_steps)
        self._pred_rewards = np.empty(hparams.num_steps)
        self._opt_rewards = np.empty(hparams.num_steps)
        self._positive_reward = hparams.positive_reward
        self._n = 0
        self._sum_cost = 0
        self._tp = self._fp = self._tn = self._fn = 0
        self._precision = np.empty(hparams.num_steps)
        self._recall = np.empty(hparams.num_steps)
        self._positives = np.empty(hparams.num_steps)

    def add(self, action_i, pred_r, opt_r):
        self._actions[self.n] = action_i
        self._pred_rewards[self.n] = pred_r
        self._opt_rewards[self.n] = opt_r
        self._n += 1
        self._sum_cost += self.cost(self.n - 1)

        opt_pos = opt_r > 0
        pred_pos = pred_r > self._positive_reward / 2
        self._tp += int(opt_pos and pred_pos)
        self._fp += int(not opt_pos and pred_pos)
        self._tn += int(not opt_pos and not pred_pos)
        self._fn += int(opt_pos and not pred_pos)

        # update precision
        if (self.tp + self.fp) == 0:
            self._precision[self.n - 1] = 0
        else:
            self._precision[self.n - 1] = self.tp / (self.tp + self.fp)

        # update recall
        if (self.tp + self.fn) == 0:
            self._recall[self.n - 1] = 0
        else:
            self._recall[self.n - 1] = self.tp / (self.tp + self.fn)

        # update positives
        self._positives[self.n - 1] = self.tp + self.fn

    def cost(self, i=None):
        if i is None:
            return np.abs(self.pred_rewards - self.opt_rewards)
        else:
            if i >= self.n or i < -self.n:
                raise ValueError('Index out of bounds.')
            return np.abs(self.pred_rewards[i] - self.opt_rewards[i])

    @property
    def actions(self):
        return self._actions[:self.n]

    @property
    def pred_rewards(self):
        return self._pred_rewards[:self.n]

    @property
    def opt_rewards(self):
        return self._opt_rewards[:self.n]

    @property
    def n(self):
        return self._n

    @property
    def mean_cost(self):
        if self.n == 0:
            raise ValueError('No historical data to calculate mean cost.')
        return self._sum_cost / self.n

    @property
    def tp(self):
        return self._tp

    @property
    def fp(self):
        return self._fp

    @property
    def tn(self):
        return self._tn

    @property
    def fn(self):
        return self._fn

    @property
    def precision(self):
        return self._precision[:self.n]

    @property
    def recall(self):
        return self._recall[:self.n]

    @property
    def positives(self):
        return self._positives
