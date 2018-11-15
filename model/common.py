import tensorflow as tf



def get_optimizer(start_lr=0.0001,
                  every_step=100,
                  coef=0.5):
    def get_optimizer():
        global_step = tf.train.get_global_step()

        lr = tf.train.exponential_decay(start_lr,
                                        global_step,
                                        every_step,
                                        coef,
                                        staircase=True)
        lr_ = tf.Variable(lr,
                          dtype=tf.float32,
                          trainable=False)
        opt_D = tf.train.AdamOptimizer(lr_)
        return opt_D

    return get_optimizer
