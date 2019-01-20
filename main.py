import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights, unlinkWeights, copyWeights
from data.gestalt_shapes_1 import load_dataset
from run_output import dump_images
from loss import loss

image_shape = (64, 64)

label_channels = (
    'edges',
    'filled',
    'circles',
    'squares',
    'triangles'
)

allDataAndLabels = load_dataset(dtype=tf.float16,
                                data_shape=image_shape,
                                label_shape=image_shape,
                                label_channels=label_channels
                                )

# validation_subset = allDataAndLabels.take(20)

# load old model first so that the new one can initialize properly
# oldModel = keras.models.load_model("garnet_r15.h5", compile=False)
# oldModel.load_weights("./saved_models/garnet_r15")

# build new model
model = generateModel((image_shape[0], image_shape[1], 2),
                      output_filters=5,
                      initial_filters=8,
                      logic_filters=32,
                      kernel_size=7,
                      rec_depth=40,
                      prefix=''
                      )

with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())

    # model = keras.models.load_model("", compile=False)
    model.load_weights("./saved_models/garnet_r19", by_name=True)

    # link repeated layers
    linkWeights(model, offset=2)

    # summarize and compile model
    model.summary()
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    # copy weights from previous run
    # numCopied = copyWeights(sess, oldModel, model, '', '')
    # print("copied {0} weights from old model".format(numCopied))

    # set up monitoring
    tensorboard = keras.callbacks.TensorBoard(log_dir="./graph",
                                              histogram_freq=0,
                                              write_graph=True,
                                              write_images=True,
                                              write_grads=True
                                              )

    # do the math
    model.fit(allDataAndLabels,
              epochs=200,
              steps_per_epoch=50,
              callbacks=[tensorboard]
              )

    # unlink layers prior to saving
    unlinkWeights(model, sess)

    # save the model
    model.save_weights('./saved_models/garnet_r21', save_format='h5')
    model.save("garnet_r21.h5")

    # save output samples and exit
    dump_images(sess, model, allDataAndLabels, 100, 0.3,
                channels=label_channels)
    exit()
