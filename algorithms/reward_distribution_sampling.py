import numpy as np

from core.bandit_algorithm import BanditAlgorithm
from core.summary_writer import SummaryWriter
from algorithms.neural_bandit_model import NeuralBanditModel


class RewardDistributionSampling(BanditAlgorithm):

    def __init__(self, hparams, data):
        self.hparams = hparams
        self.data = data
        self.summary = SummaryWriter(hparams)
        self.bnn = NeuralBanditModel(optimizer='RMS', hparams=self.hparams, name=self.hparams.name)

        self._sum_cost = 0
        self._positive_samples = 0
        self._num_samples = 0

    def action(self):
        with self.bnn.graph.as_default():
            # Retrieve distribution over actions
            action_distribution, pred_rs = self.bnn.sess.run([self.bnn.distribution, self.bnn.y_pred],
                                                             feed_dict={self.bnn.x: self.data.actions})
            action_distribution = action_distribution.reshape((-1,))
            pred_rs = pred_rs.reshape((-1,))

            # Pick action randomly according to distribution
            action_i = np.random.choice(self.data.num_actions, p=action_distribution)
            opt_r = self.data.rewards[action_i]
            self.summary.add(action_i, pred_rs[action_i], opt_r)

            return action_i

    def update(self, action_i):
        with self.bnn.graph.as_default():
            # Play action, get regret and perform an update step of representation
            action, opt_r = self.data.get_row(action_i)
            self.bnn.sess.run(self.bnn.train_op,
                              feed_dict={self.bnn.x: action.reshape((1, -1)),
                                         self.bnn.y: opt_r.reshape((1, -1))})
