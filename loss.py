import tensorflow as tf

def loss(truth, prediction):
    diffLoss = tf.reduce_sum(
        tf.square(tf.subtract(
            tf.nn.selu(truth),
            tf.nn.selu(prediction)
        )),
        axis=3
    )

    catLoss = tf.losses.softmax_cross_entropy(
        truth,
        prediction,
        label_smoothing=1
        )

    return tf.add(
        diffLoss,
        catLoss
    )
