import tensorflow as tf
from tensorflow import keras
from model import generateModel
from data import small_shapes

(allData, allLabels) = small_shapes.load_data(50)

model = generateModel((64, 64, 1), output_filters = 6)
model.summary()
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(allData, allLabels, epochs=5, steps_per_epoch=10)

model.save_weights('./saved_models/garnet-r1')
