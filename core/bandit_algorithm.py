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

"""Define the abstract class for contextual bandit algorithms."""
from abc import ABC, abstractmethod
from absl import flags
from copy import deepcopy
import numpy as np

from core.historical_dataset import HistoricalDataset

FLAGS = flags.FLAGS


class BanditAlgorithm(ABC):
    """A bandit algorithm must be able to do two basic operations.

    1. Choose an action.
    2. Update its internal model given an action and it's reward.
    """

    def __init__(self, hparams, data):
        self.hparams = hparams
        self.data = deepcopy(data)

        self.data.positive_reward = self.hparams.positive_reward
        self.data.negative_reward = self.hparams.negative_reward

        self.h_data = HistoricalDataset(intercept=getattr(self.hparams, 'intercept', False))

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def update(self, action_i, action, pred_r, opt_r):
        pass

    def run(self):
        if FLAGS.verbose:
            print('Running configuration:')
            for param_key, param_value in self.hparams.values().items():
                print('\t', param_key, '=', param_value)
            print()

        for step in range(self.hparams.num_steps):
            if FLAGS.verbose and (step + 1) % FLAGS.checkpoint_freq == 0:
                print('>> Step:', step + 1)
                self.hparams.show_training = True
            else:
                self.hparams.show_training = False

            if step >= self.hparams.positive_start:
                action_i, action, pred_r, opt_r = self.action()
            else:
                action_i = step
                action = self.data.actions[step]
                pred_r = opt_r = self.data.rewards[step]
            self.update(action_i, action, pred_r, opt_r)

            if FLAGS.verbose and (step + 1) % FLAGS.checkpoint_freq == 0:
                print('Action:', action_i, ';\tPred reward:', pred_r, ';\tOpt reward:', opt_r)
                print('Actions left: {};\tPositives sampled: {};\tPositives left: {}'.format(
                    self.data.num_actions,
                    len(np.where(self.h_data.opt_rewards == self.data.positive_reward)[0]),
                    len(self.data.positive_actions)
                ))
                print()

        return self.h_data
