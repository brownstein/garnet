import tensorflow as tf
from tensorflow import keras
from model import generateModel
#from loss import loss
from data.small_shapes import load_data_as_dataset

#(allData, allLabels) = load_data_as_dataset(max_tests=25, offset=100)

#allData = tf.summary.scalar("data", allData)
#allLabels = tf.summary.scalar("labels", allLabels)

allTheThings = load_data_as_dataset(max_tests=25, offset=100)
allTheThings = allTheThings.make_one_shot_iterator()

def loss(truth, prediction):
    sum = tf.reduce_sum(tf.math.squared_difference(truth, prediction))
    return sum

model = generateModel((64, 64, 1),
                      output_filters=6,
                      logic_filters=20,
                      kernel_size=7,
                      rec_depth=4)
model.summary()
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=loss,
              metrics=['accuracy'])


tensorboard = keras.callbacks.TensorBoard(log_dir="./graph",
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

model.fit(allTheThings, epochs=10, steps_per_epoch=4, callbacks=[tensorboard])

model.save_weights('./saved_models/garnet-r2')
