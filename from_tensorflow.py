import tensorflow as tf
import os
from tensorflow import keras
from model import generateModel
from data.small_combinations import load_data

(allData, allLabels) = load_data(
    10,
    input_shape=(46, 46),
    label_shape=(38, 38)
    )

allLabels = list(tf.pad(d, tf.constant([[0,0], [0,0], [0, 1]])) for d in allLabels)

def loss(truth, prediction):
    return tf.reduce_sum(tf.math.squared_difference(truth
    , prediction))

model = keras.models.load_model('./t1/t1.h5')

model.load_weights('./t1/t1.h5', by_name=True)

with tf.Session().as_default() as sess:

    for w in model.weights:
        sess.run(w.initializer)

    for v in model.variables:
        sess.run(v.initializer)

    p = model.predict(tf.expand_dims(allData[5], axis=0), steps=1)
    s = tf.summary.image(name="out-p.png",
                         tensor=p[:,:,:,:1])

    summary = tf.summary.Summary.FromString(s.eval())
    png = summary.value[0].image.encoded_image_string

    f = open("out2.png", "wb")
    f.write(png)
    f.close()

print('done')
