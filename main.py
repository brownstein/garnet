import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights, unlinkWeights, copyWeights
from data.small_shapes_2 import load_dataset
from run_output import dump_images
from loss import loss

allDataAndLabels = load_dataset(dtype=tf.float16,
                                input_shape=(32, 32),
                                label_shape=(32, 32))

# load old model first so that the new one can initialize properly
oldModel = keras.models.load_model("garnet_r13.h5", compile=False)
oldModel.load_weights("./saved_models/garnet-r13")

# build new model
model = generateModel((32, 32, 2),
                      output_filters=6,
                      initial_filters=8,
                      logic_filters=32,
                      kernel_size=7,
                      rec_depth=32,
                      prefix=''
                      )

with tf.Session().as_default() as sess:

    # load weights from previous run
    copyWeights(oldModel, model, '', '')

    # link repeated layers
    linkWeights(model, offset=2)

    # summarize and compile model
    model.summary()
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    # set up monitoring
    tensorboard = keras.callbacks.TensorBoard(log_dir="./graph",
                                              histogram_freq=0,
                                              write_graph=True,
                                              write_images=True,
                                              write_grads=True
                                              )

    # do the math
    model.fit(allDataAndLabels, epochs=100, steps_per_epoch=50, callbacks=[tensorboard])

    # unlink layers prior to saving
    unlinkWeights(model)

    # save the model
    model.save_weights('./saved_models/garnet-r13', save_format='h5')
    model.save("garnet_r13.h5")

    # save output samples and exit
    dump_images(sess, model, allDataAndLabels, 0.25, 100)
    exit()
