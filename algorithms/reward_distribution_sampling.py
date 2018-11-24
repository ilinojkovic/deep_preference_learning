import numpy as np

from core.bandit_algorithm import BanditAlgorithm
from core.historical_dataset import HistoricalDataset
from algorithms.neural_bandit_model import NeuralBanditModel


class RewardDistributionSampling(BanditAlgorithm):

    def __init__(self, hparams, data):
        self.hparams = hparams
        self.data = data

        self.update_freq_nn = hparams.training_freq_network

        self.t = 0

        self.num_epochs = hparams.training_epochs

        self.h_data = HistoricalDataset()
        self.bnn = NeuralBanditModel(optimizer='RMS', hparams=self.hparams, name=self.hparams.name)

    def action(self):
        with self.bnn.graph.as_default():
            # Retrieve distribution over actions
            action_distribution, pred_rs = self.bnn.sess.run([self.bnn.distribution, self.bnn.y_pred],
                                                             feed_dict={self.bnn.x: self.data.actions})
            action_distribution = action_distribution.reshape((-1,))
            pred_rs = pred_rs.reshape((-1,))

            # Pick action randomly according to distribution
            action_i = np.random.choice(self.data.num_actions, p=action_distribution)
            return action_i, self.data.actions[action_i], pred_rs[action_i], self.data.rewards[action_i]

    def update(self, action_i, action, pred_r, opt_r):
        self.t += 1
        self.h_data.add(action_i, action, pred_r, opt_r)

        # Retrain the network on the original data (h_data)
        if self.t % self.update_freq_nn == 0:

            if self.hparams.reset_lr:
                self.bnn.assign_lr()
            self.bnn.train(self.h_data, self.num_epochs)
