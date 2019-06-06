import tensorflow as tf

def loss(truth, prediction):
    # return tf.losses.softmax_cross_entropy(truth, prediction)
    return tf.reduce_sum(
        tf.multiply(
            tf.square(tf.subtract(truth, prediction)),
            tf.add(tf.multiply(truth, 0.2), 0.8)
        )
    )
