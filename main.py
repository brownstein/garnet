import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkAllWeights, unlinkAllWeights
from data.gestalt_shapes_4 import load_dataset as load_gestalt
from data.variably_sized_shapes import load_dataset as load_vsized

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
    repeat=False
)

# allDataAndLabels = gestaltDataAndLabels

variablySizedDataAndLabels = load_vsized(
    dtype=tf.float16,
    data_shape=image_shape,
    label_shape=image_shape,
    label_channels=label_channels,
    repeat=True
)

allDataAndLabels = variablySizedDataAndLabels

# allDataAndLabels = gestaltDataAndLabels.concatenate(variablySizedDataAndLabels)
# allDataAndLabels = allDataAndLabels.shuffle(100).repeat()

# build new model
model = generateModel((image_shape[0], image_shape[1], 1),
                      output_filters=5,
                      initial_filters=8,
                      depth=40
                      )

with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())

    # load weights from previous run
    model.load_weights("./saved_models/garnet_rev_31_weights.h5", by_name=True)

    # link repeated layers
    linkAllWeights(model)

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
    model.fit(allDataAndLabels,
              epochs=100,
              steps_per_epoch=50,
              callbacks=[tensorboard]
              )

    # unlink layers prior to saving
    unlinkAllWeights(model, sess)

    # save the model
    model.save_weights('./saved_models/garnet_rev_32_weights.h5', save_format='h5')
    model.save("./saved_models/garnet_rev_32_full.h5")

    # dump output before weights are unlinked
    dump_images(sess, model, allDataAndLabels, 100, 0.3,
                channels=label_channels)

    # save output samples and exit
    exit()
