import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights, copyWeights
from data.small_shapes_2 import load_dataset
from run_output import dump_images
from loss import loss

allDataAndLabels = load_dataset(dtype=tf.float16,
                                input_shape=(32, 32),
                                label_shape=(32, 32))

model = generateModel((32, 32, 2),
                      output_filters=6,
                      initial_filters=8,
                      logic_filters=32,
                      kernel_size=7,
                      rec_depth=32,
                      prefix=''
                      )

linkWeights(model, offset=1)

with tf.Session().as_default() as sess:

    # I had this working in JS... might have to change loss
    # or layers. At least it runs!

    # oldModel = keras.models.load_model("garnet2.h5", compile=False)
    # oldModel.load_weights("garnet2.h5")
    # copyWeights(oldModel, model)

    model.load_weights("./saved_models/garnet-r11")

    model.summary()
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir="./graph",
                                              histogram_freq=10,
                                              write_graph=True,
                                              write_images=True,
                                              write_grads=True
                                              )

    model.fit(allDataAndLabels, epochs=20, steps_per_epoch=100, callbacks=[tensorboard])

    model.save_weights('./saved_models/garnet-r11', save_format='h5')
    model.save("garnet_r11.h5")

    dump_images(sess, model, allDataAndLabels, 0.5, 100)

    exit()
