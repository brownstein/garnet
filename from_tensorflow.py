import tensorflow as tf
import os
from tensorflow import keras
from model import generateModel
from data.small_combinations import load_data

(allData, allLabels) = load_data(
    10,
    input_shape=(44, 44),
    label_shape=(38, 38)
    )

allLabels = list(tf.pad(d, tf.constant([[0,0], [0,0], [0, 1]])) for d in allLabels)

def loss(truth, prediction):
    return tf.reduce_sum(tf.math.squared_difference(truth
    , prediction))

model = keras.models.load_model('./keras_saved/test')

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss=loss,
              metrics=['accuracy'])

model.fit(tf.stack(allData, 0), tf.stack(allLabels, 0), epochs=3, steps_per_epoch=4)

p = model.predict(tf.expand_dims(allData[0], axis=0), steps=1)

with tf.Session().as_default() as sess:
    s = tf.summary.image(name="out-p.png",
                         tensor=p[:,:,:,:3])

    summary = tf.summary.Summary.FromString(s.eval())
    png = summary.value[0].image.encoded_image_string

    f = open("out1.png", "wb")
    f.write(png)
    f.close()

print('done')
