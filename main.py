import tensorflow as tf
from tensorflow import keras
from model import generateModel
#from loss import loss
from data import small_shapes

(allData, allLabels) = small_shapes.load_data(150)

def loss(truth, preduction):
    return tf.reduce_sum(tf.math.squared_difference(truth, preduction))

model = generateModel((64, 64, 1),
                      output_filters = 6)
model.summary()
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=loss,
              metrics=['accuracy'])

model.fit(allData, allLabels, epochs=50, steps_per_epoch=20)

model.save_weights('./saved_models/garnet-r1')
