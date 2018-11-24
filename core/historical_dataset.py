import numpy as np


class HistoricalDataset(object):

    def __init__(self, intercept=False):
        self._action_indices = []
        self._actions = None
        self._pred_rewards = []
        self._opt_rewards = []
        self.intercept = intercept

    def add(self, action_i, action, pred_r, opt_r):
        if self.intercept:
            a = np.array(action[:])
            a = np.append(a, 1.0).reshape((1, -1))
        else:
            a = np.array(action[:]).reshape((1, -1))

        if self._actions is None:
            self.actions = a
        else:
            self.actions = np.vstack((self.actions, a))

        self._action_indices.append(action_i)
        self._pred_rewards.append(pred_r)
        self._opt_rewards.append(opt_r)

    def get_batch(self, batch_size):
        ind = np.random.choice(len(self.actions), batch_size)
        return self.actions[ind, :], self.opt_rewards[ind].reshape((-1, 1))

    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, value):
        self._actions = value

    @property
    def action_indices(self):
        return np.array(self._action_indices)

    @property
    def pred_rewards(self):
        return np.array(self._pred_rewards)

    @property
    def opt_rewards(self):
        return np.array(self._opt_rewards)

    @property
    def n(self):
        return len(self.actions)
