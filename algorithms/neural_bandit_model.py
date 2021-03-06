"""Define a family of neural network architectures for bandit   s.

The network accepts different type of optimizers that could lead to different
approximations of the posterior distribution or simply to point estimates.
"""
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS


class NeuralBanditModel(object):
    """Implements a neural network for bandit problems."""

    def __init__(self, optimizer, hparams, name):
        """Saves hyper-params and builds the Tensorflow graph."""

        self.opt_name = optimizer
        self.name = name
        self.hparams = hparams
        self.verbose = getattr(self.hparams, "verbose", True)
        self.times_trained = 0
        self.build_model()

    def build_layer(self, x, num_units):
        """Builds a layer with input x; dropout and layer norm if specified."""

        init_s = self.hparams.init_scale

        layer_n = getattr(self.hparams, "layer_norm", False)
        dropout = getattr(self.hparams, "use_dropout", False)

        if self.hparams.activation.lower() == 'sigmoid':
            activation = tf.nn.sigmoid
        elif self.hparams.activation.lower() == 'tanh':
            activation = tf.nn.tanh
        elif self.hparams.activation.lower() == 'none':
            activation = None
        else:
            activation = tf.nn.relu

        nn = tf.contrib.layers.fully_connected(
            x,
            num_units,
            activation_fn=activation,
            normalizer_fn=None if not layer_n else tf.contrib.layers.layer_norm,
            normalizer_params={},
            weights_initializer=tf.random_uniform_initializer(-init_s, init_s)
        )

        if dropout:
            nn = tf.nn.dropout(nn, self.hparams.keep_prob)

        return nn

    def forward_pass(self):

        init_s = self.hparams.init_scale

        scope_name = "prediction_{}".format(self.name)
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            nn = self.x
            if self.hparams.layer_sizes:
                for num_units in self.hparams.layer_sizes:
                    if num_units > 0:
                        nn = self.build_layer(nn, num_units)

            y_pred = tf.contrib.layers.fully_connected(
                nn,
                1,
                activation_fn=tf.sigmoid,
                weights_initializer=tf.random_uniform_initializer(-init_s, init_s)
            )

            y_pred = self.hparams.positive_reward * y_pred

        return nn, y_pred

    def build_model(self):
        """Defines the actual NN model with fully connected layers.

        The loss is computed for partial feedback settings (bandits), so only
        the observed outcome is backpropagated (see weighted loss).
        Selects the optimizer and, finally, it also initializes the graph.
        """

        # create and store the graph corresponding to the BNN instance
        self.graph = tf.Graph()

        with self.graph.as_default():

            # create and store a new session for the graph
            self.sess = tf.Session()

            with tf.name_scope(self.name):

                self.global_step = tf.train.get_or_create_global_step()

                # context
                self.x = tf.placeholder(
                    shape=[None, self.hparams.actions_dim],
                    dtype=tf.float32,
                    name="{}_x".format(self.name))

                # reward vector
                self.y = tf.placeholder(
                    shape=[None, 1],
                    dtype=tf.float32,
                    name="{}_y".format(self.name))

                # with tf.variable_scope("prediction_{}".format(self.name)):
                self.nn, self.y_pred = self.forward_pass()
                self.distribution = self.y_pred / tf.reduce_sum(self.y_pred)
                self.cost = self.get_cost()

                if self.hparams.activate_decay:
                    self.lr = tf.train.inverse_time_decay(
                        self.hparams.initial_lr, self.global_step,
                        1, self.hparams.lr_decay_rate)
                else:
                    self.lr = tf.Variable(self.hparams.initial_lr, trainable=False)

                tvars = tf.trainable_variables()
                grads = tf.gradients(self.cost, tvars)
                norm = tf.global_norm(grads)
                norm = tf.where(tf.logical_or(tf.is_nan(norm), tf.is_inf(norm)), tf.zeros_like(norm), norm)

                grads, _ = tf.clip_by_global_norm(grads, self.hparams.max_grad_norm, use_norm=norm)

                self.optimizer = self.select_optimizer()

                self.train_op = self.optimizer.apply_gradients(
                    zip(grads, tvars), global_step=self.global_step)

                # create tensorboard metrics
                self.create_summaries()
                self.summary_writer = tf.summary.FileWriter(
                    "{}/graph_{}".format(FLAGS.logdir, self.name), self.sess.graph)

                self.init = tf.global_variables_initializer()

                self.initialize_graph()

    def initialize_graph(self):
        """Initializes all variables."""

        with self.graph.as_default():
            if self.verbose:
                print("Initializing model {}.".format(self.name))
            self.sess.run(self.init)

    def get_cost(self):
        return tf.losses.absolute_difference(self.y_pred, self.y, reduction=tf.losses.Reduction.MEAN)

    def assign_lr(self):
        """Reses the learning rate in dynamic schedules for subsequent trainings.

        In bandits settings, we do expand our dataset over time. Then, we need to
        re-train the network with the new data. The algorithms that do not keep
        the step constant, can reset it at the start of each *training* process.
        """

        decay_steps = 1
        if self.hparams.activate_decay:
            current_gs = self.sess.run(self.global_step)
            with self.graph.as_default():
                self.lr = tf.train.inverse_time_decay(self.hparams.initial_lr,
                                                      self.global_step - current_gs,
                                                      decay_steps,
                                                      self.hparams.lr_decay_rate)

    def select_optimizer(self):
        """Selects optimizer. To be extended (SGLD, KFAC, etc)."""
        return tf.train.RMSPropOptimizer(self.lr)

    def create_summaries(self):
        """Defines summaries including mean loss, learning rate, and global step."""

        with self.graph.as_default():
            with tf.name_scope(self.name + "_summaries"):
                tf.summary.scalar("cost", self.cost)
                tf.summary.scalar("lr", self.lr)
                tf.summary.scalar("global_step", self.global_step)
                self.summary_op = tf.summary.merge_all()

    def train(self, data, num_steps):
        """Trains the network for num_steps, using the provided data.

        Args:
          data: SummaryData object that provides the data.
          num_steps: Number of minibatches to train the network for.
        """

        if self.hparams.show_training:
            print("Training {} for {} steps...".format(self.name, num_steps))

        with self.graph.as_default():

            for step in range(num_steps):
                x, y = data.get_batch(self.hparams.batch_size)
                _, cost, summary, lr = self.sess.run(
                    [self.train_op, self.cost, self.summary_op, self.lr],
                    feed_dict={self.x: x, self.y: y})

                if (step + 1) % self.hparams.freq_summary == 0:
                    if self.hparams.show_training:
                        print("{} | NN | step: {}, lr: {}, loss: {}".format(
                            self.name, step, lr, cost))
                    self.summary_writer.add_summary(summary, step)

            self.times_trained += 1
