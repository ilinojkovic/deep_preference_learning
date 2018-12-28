# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Thompson Sampling with linear posterior over a learnt deep representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from algorithms.neural_bandit_model import NeuralBanditModel
from core.bandit_algorithm import BanditAlgorithm
from core.historical_dataset import HistoricalDataset


class NeuralLinearPosteriorSampling(BanditAlgorithm):
    """Full Bayesian linear regression on the last layer of a deep neural net."""

    def __init__(self, hparams, data):
        """
        Args:
          hparams: Hyper-parameters of the algorithm.
          data: BanditDataset object containing all the actions and rewards
        """
        super().__init__(hparams, data)
        self.latent_dim = self.hparams.layer_sizes[-1]

        # Gaussian prior for each beta_i
        self._lambda_prior = self.hparams.lambda_prior

        self.mu = np.zeros(self.latent_dim)

        self.cov = (1.0 / self.lambda_prior) * np.eye(self.latent_dim)

        self.precision = self.lambda_prior * np.eye(self.latent_dim)

        # Inverse Gamma prior for each sigma2_i
        self._a0 = self.hparams.a0
        self._b0 = self.hparams.b0

        self.a = self._a0
        self.b = self._b0

        # Regression and NN Update Frequency
        self.update_freq_lr = hparams.training_freq
        self.update_freq_nn = hparams.training_freq_network

        self.t = 0

        self.num_epochs = hparams.training_epochs
        self.h_latent = HistoricalDataset()
        self.bnn = NeuralBanditModel(optimizer='RMS', hparams=self.hparams, name=self.hparams.name)

    def action(self):
        """Samples beta's from posterior, and chooses best action accordingly."""

        if len(self.data.positive_actions) == 0:
            # Return fake positive action if run out of positives
            return -1, np.zeros(self.data.actions_dim), self.data.positive_reward, self.data.positive_reward

        # Sample sigma2, and beta conditional on sigma2
        sigma2_s = self.b * invgamma.rvs(self.a)

        try:
            beta_s = np.random.multivariate_normal(self.mu, sigma2_s * self.cov)
        except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            print('Exception when sampling for {}.'.format(self.hparams.name))
            print('Details: {} | {}.'.format(str(e), e.args))
            d = self.latent_dim
            beta_s = np.random.multivariate_normal(np.zeros(d), np.eye(d))

        # Compute last-layer representation for the current context
        with self.bnn.graph.as_default():
            z_actions = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: self.data.actions})

        # Apply Thompson Sampling to last-layer representation
        pred_rs = np.dot(beta_s, z_actions.T)

        action_i = np.argmax(pred_rs)
        return action_i, self.data.actions[action_i], pred_rs[action_i], self.data.rewards[action_i]

    def update(self, action_i, action, pred_r, opt_r):
        """Updates the posterior using linear bayesian regression formula."""

        self.t += 1
        self.h_data.add(action_i, action, pred_r, opt_r)

        if action_i == -1:
            # No updates on fake action
            return

        z_action = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: self.data.actions[action_i].reshape((1, -1))})
        self.h_latent.add(action_i, z_action, pred_r, opt_r)

        if self.hparams.remove_actions:
            self.data.remove(action_i)

        # Retrain the network on the original data (data_h)
        if self.t % self.update_freq_nn == 0:

            if self.hparams.reset_lr:
                self.bnn.assign_lr()
            self.bnn.train(self.h_data, self.num_epochs)

            # Update the latent representation of every datapoint collected so far
            self.h_latent.actions = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: self.h_data.actions})

        # Update the Bayesian Linear Regression
        if self.t % self.update_freq_lr == 0:

            # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
            z = self.h_latent.actions
            y = self.h_latent.opt_rewards

            # The algorithm could be improved with sequential formulas (cheaper)
            s = np.dot(z.T, z)

            # Some terms are removed as we assume prior mu_0 = 0.
            self.precision = s + self.lambda_prior * np.eye(self.latent_dim)
            self.cov = np.linalg.inv(self.precision)
            diagonalize = getattr(self.hparams, 'diagonalize', False)
            if diagonalize:
                if diagonalize == 'precision':
                    self.precision = np.diag(np.diag(self.precision))
                    self.cov = np.linalg.inv(self.precision)
                elif diagonalize == 'covariance':
                    self.cov = np.diag(np.diag(self.cov))
                    self.precision = np.linalg.inv(self.cov)
                else:
                    raise ValueError('Wrong diagonalization parameter. Only \'precision\' and \'covariance\' supported.')
            self.mu = np.dot(self.cov, np.dot(z.T, y))

            # Inverse Gamma posterior update
            self.a = self.a0 + z.shape[0] / 2.0
            b_upd = 0.5 * np.dot(y.T, y)
            b_upd -= 0.5 * np.dot(self.mu.T, np.dot(self.precision, self.mu))
            self.b = self.b0 + b_upd

    def trainable_parameters(self):
        pass

    def reset_trainable_parameters(self, params):
        pass

    @property
    def a0(self):
        return self._a0

    @property
    def b0(self):
        return self._b0

    @property
    def lambda_prior(self):
        return self._lambda_prior
