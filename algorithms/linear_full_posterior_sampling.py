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

"""Contextual algorithm that keeps a full linear posterior for each arm."""
import numpy as np
from scipy.stats import invgamma

from core.bandit_algorithm import BanditAlgorithm


class LinearFullPosteriorSampling(BanditAlgorithm):
    """Thompson Sampling with independent linear models and unknown noise var."""

    def __init__(self, hparams, data):
        """Initialize posterior distributions and hyperparameters.

        Assume a linear model for each action i: reward = context^T beta_i + noise
        Each beta_i has a Gaussian prior (lambda parameter), each sigma2_i (noise
        level) has an inverse Gamma prior (a0, b0 parameters). Mean, covariance,
        and precision matrices are initialized, and the ContextualDataset created.

        Args:
          hparams: Hyper-parameters of the algorithm.
          data: BanditDataset object containing all the actions and rewards
        """
        super().__init__(hparams, data)

        # Gaussian prior for each beta_i
        self._lambda_prior = self.hparams.lambda_prior

        self.mu = np.zeros(self.hparams.actions_dim + 1)
        self.precision = self.lambda_prior * np.eye(self.hparams.actions_dim + 1)

        # Inverse Gamma prior for each sigma2_i
        self._a0 = self.hparams.a0
        self._b0 = self.hparams.b0

        self.a = np.array([self._a0])
        self.b = np.array([self._b0])

        self.t = 0

    def action(self):
        """Samples beta's from posterior, and chooses best action accordingly.

        Returns:
          action: Selected action for the context.
        """

        if len(self.data.positive_actions) == 0:
            # Return fake positive action if run out of positives
            return -1, np.zeros(self.data.actions_dim), self.data.positive_reward, self.data.positive_reward

        # Sample sigma2, and beta conditional on sigma2
        sigma2_s = self.b * invgamma.rvs(self.a)

        try:
            beta_s = np.random.multivariate_normal(self.mu, sigma2_s * self.cov)
        except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            print('Exception when sampling from {}.'.format(self.hparams.name))
            print('Details: {} | {}.'.format(str(e), e.args))
            d = self.hparams.actions_dim + 1
            beta_s = np.random.multivariate_normal(np.zeros(d), np.eye(d))

        # Compute sampled expected rewards, intercept is last component of beta
        pred_rs = [
            np.dot(beta_s[:-1], action.T) + beta_s[-1]
            for action in self.data.actions
        ]

        action_i = np.argmax(pred_rs)
        return action_i, self.data.actions[action_i], pred_rs[action_i], self.data.rewards[action_i]

    def update(self, action_i, action, pred_r, opt_r):
        """Updates action posterior using the linear Bayesian regression formula.

        Args:
          action_i: Last observed action index.
        """

        self.t += 1
        self.h_data.add(action_i, action, pred_r, opt_r)

        if action_i == -1:
            # No updates on fake action
            return

        if self.hparams.remove_actions:
            self.data.remove(action_i)

        # Update posterior of action with formulas: \beta | x,y ~ N(mu_q, cov_q)
        x = self.h_data.actions
        y = self.h_data.opt_rewards

        # The algorithm could be improved with sequential update formulas (cheaper)
        s = np.dot(x.T, x)

        # Some terms are removed as we assume prior mu_0 = 0.
        self.precision = s + self.lambda_prior * np.eye(self.hparams.actions_dim + 1)
        self.mu = np.dot(self.cov, np.dot(x.T, y))

        # Inverse Gamma posterior update
        self.a = np.array([self.a0 + x.shape[0] / 2.0])
        b_upd = 0.5 * (np.dot(y.T, y) - np.dot(self.mu.T, np.dot(self.precision, self.mu)))
        self.b = np.array([self.b0 + b_upd])

    def trainable_parameters(self):
        return {
            'mu': self.mu,
            'precision': self.precision,
            'a': self.a,
            'b': self.b
        }

    def reset_trainable_parameters(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    @property
    def a0(self):
        return self._a0

    @property
    def b0(self):
        return self._b0

    @property
    def lambda_prior(self):
        return self._lambda_prior

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        self._precision = value
        self._cov = np.linalg.inv(self.precision)

    @property
    def cov(self):
        return self._cov
