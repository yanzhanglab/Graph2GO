import tensorflow as tf

class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm, lr):

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds, targets=labels, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds, 0.0), tf.int32),
                                           tf.cast(tf.greater_equal(labels, 0.0),tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, lr):

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds, targets=labels, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds, 0.0), tf.int32),
                                           tf.cast(tf.greater_equal(labels, 0.0), tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
