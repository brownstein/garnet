import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights, unlinkWeights, copyWeights
from data.gestalt_shapes_2 import load_dataset as load_gestalt
# from data.variably_sized_shapes import load_dataset as load_vsized

from run_output import dump_images
from loss import loss

image_shape = (64, 64)

label_channels = (
    'fill',
    'edges',
    'square',
    'circle',
    'triangle'
)

gestaltDataAndLabels = load_gestalt(
    dtype=tf.float16,
    data_shape=image_shape,
    label_shape=image_shape,
    label_channels=label_channels,
    include_gestalt_shapes=True,
    repeat=True
)
# variablySizedDataAndLabels = load_vsized(
#     dtype=tf.float16,
#     data_shape=image_shape,
#     label_shape=image_shape,
#     label_channels=label_channels,
#     repeat=False
# )
# allDataAndLabels = gestaltDataAndLabels.concatenate(variablySizedDataAndLabels)
# allDataAndLabels = allDataAndLabels.shuffle(400).repeat()

allDataAndLabels = gestaltDataAndLabels

# build new model
model = generateModel((image_shape[0], image_shape[1], 1),
                      output_filters=5,
                      initial_filters=8,
                      logic_filters=32,
                      kernel_size=7,
                      depth=64,
                      num_variations=2,
                      extra_conv2d_props = (
                        {},
                        { dilation_rate: 3 }
                      )
                      )

with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())

    # model = keras.models.load_model("", compile=False)
    # model.load_weights("./saved_models/garnet_rev_29_weights.h5", by_name=True)

    # link repeated layers
    linkWeights(model, offset=3, targetLayersWithPrefix='repeated_')
    linkWeights(model, offset=3, targetLayersWithPrefix='repeatedDilation_')

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
              epochs=1000,
              steps_per_epoch=30,
              callbacks=[tensorboard]
              )

    # unlink layers prior to saving
    unlinkWeights(model, sess, targetLayersWithPrefix='repeated_')
    unlinkWeights(model, sess, targetLayersWithPrefix='repeatedDilation_')

    # save the model
    model.save_weights('./saved_models/garnet_rev_29_weights.h5', save_format='h5')
    model.save("./saved_models/garnet_rev_29_full.h5")

    # dump output before weights are unlinked
    dump_images(sess, model, allDataAndLabels, 100, 0.3,
                channels=label_channels)

    # save output samples and exit
    exit()
