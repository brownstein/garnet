import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights, unlinkWeights, copyWeights
# from data.small_shapes_2 import load_dataset
# from data.small_combinations import load_dataset
from data.combined_shapes_2 import load_dataset
from run_output import dump_images
from loss import loss

image_shape = (64, 64)

label_channels = (
    'edges',
    'filled',
    'filled_circle',
    'filled_square',
    'filled_triangle'
)

allDataAndLabels = load_dataset(dtype=tf.float16,
                                data_shape=image_shape,
                                label_shape=image_shape,
                                label_channels=label_channels,
                                interleve_data_cases=True
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
                      rec_depth=60,
                      prefix=''
                      )

with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())

    # model = keras.models.load_model("", compile=False)
    model.load_weights("./saved_models/garnet_r19C", by_name=True)

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
              epochs=2,
              steps_per_epoch=50,
              callbacks=[tensorboard]
              )

    # unlink layers prior to saving
    unlinkWeights(model, sess)

    # save the model
    model.save_weights('./saved_models/garnet_r19C', save_format='h5')
    model.save("garnet_r19C.h5")

    # save output samples and exit
    dump_images(sess, model, allDataAndLabels, 100, 0.3,
                channels=label_channels)
    exit()
