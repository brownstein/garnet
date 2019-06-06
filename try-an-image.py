import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights
from loss import loss

image_shape = (383, 500)

model = generateModel((image_shape[0], image_shape[1], 1),
                      output_filters=5,
                      initial_filters=8,
                      depth=8
                      )

with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())

    # model = keras.models.load_model("./saved_models/garnet_rev_39_full.h5", compile=False)
    model.load_weights("./saved_models/garnet_rev_39_weights.h5", by_name=True)

    # link repeated layers
    # linkWeights(model, offset=2)

    # summarize and compile model
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    imageString = tf.read_file("bg_test.jpg")
    imageTensor = tf.image.decode_jpeg(imageString, 3)
    imageTensor = tf.cast(imageTensor, tf.float16)
    imageTensor = tf.image.resize_images(imageTensor, image_shape)
    imageTensor = imageTensor[:, :, 0:1]
    imageTensor = tf.expand_dims(imageTensor, 0)

    p = model.predict(imageTensor, steps=1)

    summaryImage = tf.summary.image(tensor=p[:,:,:,0:3], name="tester.png")
    summary = tf.summary.Summary.FromString(summaryImage.eval())
    png = summary.value[0].image.encoded_image_string
    f = open("bg_output.png", "wb")
    f.write(png)
    f.close()

    exit()
