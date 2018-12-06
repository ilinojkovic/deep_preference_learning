"""Define dataset wrapper object for easier and cleaner access to actions and rewards"""
import numpy as np


class BanditDataset(object):

    def __init__(self, actions, rewards, positive_start=0):
        """Creates BanditDataset object.

        Data is stored in attributes: actions and rewards

        Args:
            actions: Numpy array of shape [n, d], where n is number of all possible actions,
                     and d the dimension of each action
            rewards: Numpy array of shape [n,], each entry is reward for corresponding action
            positive_start: Number of positive examples to prepend to the beginning

        """

        if len(actions) != len(rewards):
            raise ValueError('Number of actions and rewards doesn\'t match.')

        if len(actions) != actions.shape[0]:
            raise ValueError('len(actions) = {}; actions.shape[0] = {}'.format(len(actions), actions.shape[0]))

        self._asd = [0, 0, 0]
        self._actions = actions
        self._rewards = rewards
        self._order = np.arange(self.num_actions)
        self._positive_start = positive_start
        self.positive_reward = 1
        self.negative_reward = -1

        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self._order)

        if self._positive_start > 0:
            # Put some positives in front
            first_positives = np.random.choice(self.positive_actions, size=self._positive_start, replace=False)
            rest = np.delete(np.arange(self.num_actions), first_positives)
            reorder = np.concatenate((first_positives, rest))
            self._order = self._order[reorder]

    def get_row(self, index):
        """Returns (action, reward) tuple for specified index"""
        return self.actions[index], self.rewards[index]

    def remove(self, index):
        """Removes action and reward in specified location"""
        underlying_index = self._order[index]
        self._actions = np.delete(self._actions, underlying_index, axis=0)
        self._rewards = np.delete(self._rewards, underlying_index)

        self._order = np.delete(self._order, index)
        self._order[self._order > underlying_index] -= 1

    @property
    def actions_dim(self):
        return self._actions.shape[1]

    @property
    def num_actions(self):
        return self._actions.shape[0]

    @property
    def actions(self):
        return self._actions[self._order]

    @actions.setter
    def actions(self, value):
        self._actions = value

    @property
    def rewards(self):
        return self._rewards[self._order]

    @property
    def positive_reward(self):
        return self._positive_reward

    @positive_reward.setter
    def positive_reward(self, value):
        self._positive_reward = value
        self._rewards[self._order[self.positive_actions]] = value

    @property
    def negative_reward(self):
        return self._negative_reward

    @negative_reward.setter
    def negative_reward(self, value):
        self._negative_reward = value
        self._rewards[self._order[self.negative_actions]] = value

    @property
    def positive_actions(self):
        return np.where(self.rewards > 0)[0]

    @property
    def negative_actions(self):
        return np.where(self.rewards <= 0)[0]
