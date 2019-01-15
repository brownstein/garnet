import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights
from data.small_combinations import load_data_as_dataset
from run_output import run_output_summaries

allDataAndLabels = load_data_as_dataset()

def loss(truth, prediction):
    return tf.reduce_sum(
        tf.math.squared_difference(truth, prediction)
        )

model = generateModel((64, 64, 2),
                      output_filters=6,
                      logic_filters=20,
                      kernel_size=7,
                      rec_depth=10)

linkWeights(model)

with tf.Session().as_default() as sess:
    # model.load_weights("./saved_models/garnet-r4")

    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(
                    learning_rate=0.1,
                    epsilon=0.1
                  ),
                  loss=loss,
                  metrics=['accuracy'])


    tensorboard = keras.callbacks.TensorBoard(log_dir="./graph",
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    model.fit(allDataAndLabels, epochs=100, steps_per_epoch=25, callbacks=[tensorboard])

    model.save_weights('./saved_models/garnet-r5')
    model.save("garnet.h5")

    run_output_summaries(sess, model, allDataAndLabels)
