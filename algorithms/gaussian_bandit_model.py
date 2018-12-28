import tensorflow as tf

from algorithms.neural_bandit_model import NeuralBanditModel


class GaussianBanditModel(NeuralBanditModel):
    """
    Extension of NeuralBanditModel. Implements a neural network
    for bandit problems, but as the output estimated the mean
    and the variance of the reward. It draws then the reward
    from Gaussian distribution using the predicted mean and
    variance.
    """

    def forward_pass(self):
        init_s = self.hparams.init_scale

        scope_name = "prediction_{}".format(self.name)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            nn = self.x
            if self.hparams.layer_sizes:
                for num_units in self.hparams.layer_sizes:
                    if num_units > 0:
                        nn = self.build_layer(nn, num_units)

            self.mu = tf.contrib.layers.fully_connected(
                nn,
                1,
                activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(-init_s, init_s)
            )

            self.log_sigma = tf.contrib.layers.fully_connected(
                nn,
                1,
                activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(-init_s, init_s)
            )

            self.log_sigma += 1e-6

            eps = tf.random_normal(tf.shape(self.log_sigma))
            y_pred = self.mu + eps * tf.exp(self.log_sigma)

        return nn, y_pred

    def get_cost(self):
        regret = tf.losses.absolute_difference(self.y_pred, self.y, reduction=tf.losses.Reduction.MEAN)

        if getattr(self.hparams, 'kl', False):
            kl_loss = tf.reduce_mean(
                - 0.5 * tf.reduce_sum(1 + self.log_sigma - tf.square(self.mu) - tf.square(tf.exp(self.log_sigma)), 1))
        else:
            kl_loss = tf.zeros_like(regret)

        return 0.005 * kl_loss + regret
