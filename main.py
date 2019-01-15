import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights
from data.small_combinations import load_data_as_dataset
from run_output import run_output_summaries

allDataAndLabels = load_data_as_dataset(dtype=tf.float16)

def loss(truth, prediction):
    return tf.reduce_sum(
        tf.math.squared_difference(truth, prediction)
        )

model = generateModel((64, 64, 2),
                      output_filters=6,
                      initial_filters=8,
                      logic_filters=32,
                      kernel_size=7,
                      rec_depth=8)

linkWeights(model)

with tf.Session().as_default() as sess:

    # model.load_weights("./saved_models/garnet-r4")

    model.summary()
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])


    tensorboard = keras.callbacks.TensorBoard(log_dir="./graph",
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    model.fit(allDataAndLabels, epochs=1000, steps_per_epoch=25, callbacks=[tensorboard])

    model.save_weights('./saved_models/garnet-r5')
    model.save("garnet.h5")

    run_output_summaries(sess, model, allDataAndLabels, 0.1)
