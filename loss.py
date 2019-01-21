import tensorflow as tf

def loss(truth, prediction):
    return tf.losses.huber_loss(truth, prediction)
    # return tf.losses.mean_squared_error(truth, prediction)
