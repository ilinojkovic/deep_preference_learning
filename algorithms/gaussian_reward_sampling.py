import numpy as np

from core.bandit_algorithm import BanditAlgorithm
from algorithms.gaussian_bandit_model import GaussianBanditModel


class GaussianRewardSampling(BanditAlgorithm):
    """
    Models the uncertainty of predicted reward by treating it as
    a random variable: r ~ N(mu, var)
    Parameters mu and var are estimated by a neural network.
    """

    def __init__(self, hparams, data):
        super().__init__(hparams, data)

        self.update_freq_nn = hparams.training_freq_network

        self.t = 0

        self.num_epochs = hparams.training_epochs

        self.bnn = GaussianBanditModel(optimizer='RMS', hparams=self.hparams, name=self.hparams.name)

    def action(self):

        if len(self.data.positive_actions) == 0:
            # Return fake positive action if run out of positives
            return -1, np.zeros(self.data.actions_dim), self.data.positive_reward, self.data.positive_reward

        with self.bnn.graph.as_default():
            # Retrieve distribution over actions
            pred_rs = self.bnn.sess.run(self.bnn.y_pred, feed_dict={self.bnn.x: self.data.actions})

            # Pick action randomly according to distribution
            action_i = np.argmax(pred_rs)
            return action_i, self.data.actions[action_i], pred_rs[action_i, 0], self.data.rewards[action_i]

    def update(self, action_i, action, pred_r, opt_r):
        self.t += 1
        self.h_data.add(action_i, action, pred_r, opt_r)

        if action_i == -1:
            # No updates on fake action
            return

        if self.hparams.remove_actions:
            self.data.remove(action_i)

        # Retrain the network on the original data (h_data)
        if self.t % self.update_freq_nn == 0:

            if self.hparams.reset_lr:
                self.bnn.assign_lr()
            self.bnn.train(self.h_data, self.num_epochs)

    def trainable_parameters(self):
        pass

    def reset_trainable_parameters(self, params):
        pass
