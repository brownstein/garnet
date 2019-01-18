import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights
from loss import loss

# build new model
model = generateModel((276, 361, 2),
                      output_filters=5,
                      initial_filters=8,
                      logic_filters=32,
                      kernel_size=7,
                      rec_depth=80,
                      prefix=''
                      )

with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())

    # model = keras.models.load_model("garnet_r16.h5", compile=False)
    model.load_weights("garnet_r17.h5", by_name=True)

    # link repeated layers
    linkWeights(model, offset=2)

    # summarize and compile model
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    imageString = tf.read_file("bg_test.jpg")
    imageTensor = tf.image.decode_jpeg(imageString, 3)
    imageTensor = tf.cast(imageTensor, tf.float16)
    imageTensor = imageTensor[:, :, 0:2]
    imageTensor = tf.expand_dims(imageTensor, 0)

    p = model.predict(imageTensor, steps=1)

    summaryImage = tf.summary.image(tensor=p[:,:,:,0:3], name="tester.png")
    summary = tf.summary.Summary.FromString(summaryImage.eval())
    png = summary.value[0].image.encoded_image_string
    f = open("bg_output.png", "wb")
    f.write(png)
    f.close()

    exit()
