import tensorflow as tf
from tensorflow import keras
from model import generateModel
#from loss import loss
from data.small_shapes import load_data

(allData, allLabels) = load_data(max_tests=50, offset=100)

def loss(truth, prediction):
    return tf.reduce_sum(tf.math.squared_difference(truth, prediction))

model = generateModel((64, 64, 1),
                      output_filters=6,
                      logic_filters=20,
                      kernel_size=7,
                      rec_depth=16)
model.summary()
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=loss,
              metrics=['accuracy'])

model.fit(allData, allLabels, epochs=200, steps_per_epoch=3)

model.save_weights('./saved_models/garnet-r3')
