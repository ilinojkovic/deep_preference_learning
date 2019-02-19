# Deep Preference Learning for Advanced Home Search

The code was produced for the semester project at ETH Zurich.
The project report can be found [here](https://drive.google.com/file/d/1v2Z-co1GFsnAaBMgAB6PhM9Tu-K5qHV_/view?usp=sharing).

We treat preference learning as a bandit problem, where our property
advertisements are the actions. Four algorithms are provided, which
try to estimate the reward for each action, choose a new action to present
to the user, and update the model parameters using the user feedback.
One model is trained per user, thus we can look at model parameters as
user representation.

Implemented models are:
1. **Reward Distribution Sampling** - An end-to-end neural network which
estimates the reward directly from action features. A new action is proposed
by sampling an action from the whole action set with probability proportional
to the estimated reward. The algorithm is implemented in
`reward_distribution_sampling.py`
2. **Bayesian Linear Regression** - We implement this linear approach as a
nice baseline. The algorithm is implemented in `linear_full_posterior_sampling.py`.
3. **Neural Linear Sampling** - To increase the representational power of
the linear algorithm, a combination of the previous two model is
implemented, where the output of the last layer of the neural network is
passed as an input to the Bayesian Linear model. The algorithm implementation
can be found in `neural_linear_sampling.py`.
4. **Gaussian Reward Sampling** - Uncertainty of the reward estimate
was taken into account by treating the reward itself as a random variable.
Therefore we proposed a neural network model which estimates mu and sigma,
and then samples the reward `r ~ N(mu, sigma)`. Action maximizing this reward
was proposed in each step. The algorithm is implemented in `gaussian_reward_sampling.py`.


## Running the models
Basic hyper-parameter grid search was also implemented. An example of
running the models as well as parameter gridding can be found in `grid.py`.
Grid search is specified in the hyper-parameter configuration dictionary,
where keys are parameter names, and values are the values of the parameters.
If the value is a list, a new tf.contrib.training.HParams is created for each
entry in the list.

## Meta-leaning
To reduce the number of steps needed for an algorithm to converge, we investigated
the [Baldwinian Meta-learning algorithm](https://arxiv.org/abs/1806.07917). Our
implementation of this algorithm can be found in `genetic_algorithm.py`. An example
script of running the algorithm can be found in `meta.py`.

## Remarks
We used and modified some algorithm implementations from [Deep Bayesian Bandits Library](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits),
and also took some inspiration from the paper of the same authors, [Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling](https://arxiv.org/abs/1802.09127).
