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


class BanditAlgorithm(object):
    """A bandit algorithm must be able to do two basic operations.

    1. Choose an action.
    2. Update its internal model given an action and it's reward.
    """

    def action(self):
        pass

    def update(self, action_i):
        pass
