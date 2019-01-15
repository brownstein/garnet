import tensorflow as tf
import os
from tensorflow import keras
from model import generateModel
from data.small_shapes import load_data_as_dataset

dataSet = load_data_as_dataset()
dataIt = dataSet.make_one_shot_iterator()
dataIt.get_next()
dataIt.get_next()
dataIt.get_next()
dataIt.get_next()
dataIt.get_next()
dataIt.get_next()
[testCase, _] = dataIt.get_next()

print(testCase)

with tf.Session().as_default() as sess:

    model = keras.models.load_model("garnet.h5")
    model.load_weights("garnet.h5")

    for w in model.weights:
        sess.run(w.initializer)

    for v in model.variables:
        sess.run(v.initializer)

    # predict an image
    p = model.predict(testCase, steps=1)
    s = tf.summary.image(name="out-p.png",
                         tensor=p[:,:,:,:1])

    # write the image
    summary = tf.summary.Summary.FromString(s.eval())
    png = summary.value[0].image.encoded_image_string
    f = open("out2.png", "wb")
    f.write(png)
    f.close()

print('done')
