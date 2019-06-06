import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkAllWeights, unlinkAllWeights, copyWeights
from data.gestalt_shapes_4 import load_dataset as load_gestalt
# from data.variably_sized_shapes import load_dataset as load_vsized
from data.combined_shapes_3 import load_dataset

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
#
# variablySizedDataAndLabels = load_vsized(
#     dtype=tf.float16,
#     data_shape=image_shape,
#     label_shape=image_shape,
#     label_channels=label_channels,
#     repeat=False
# )
# # allDataAndLabels = variablySizedDataAndLabels
#
# allDataAndLabels = gestaltDataAndLabels.concatenate(variablySizedDataAndLabels)
# allDataAndLabels = allDataAndLabels.shuffle(100).repeat()

filledData = load_dataset(
    dtype=tf.float16,
    data_shape=image_shape,
    label_shape=image_shape,
    label_channels=label_channels,
    data_channels=('fill',),
    repeat=False
)

# edgesData = load_dataset(
#     dtype=tf.float16,
#     data_shape=image_shape,
#     label_shape=image_shape,
#     label_channels=label_channels,
#     data_channels=('edges',),
#     repeat=False
# )

allDataAndLabels = filledData #.concatenate(edgesData)
allDataAndLabels = allDataAndLabels.concatenate(gestaltDataAndLabels)
allDataAndLabels = allDataAndLabels.shuffle(50).repeat()

# oldModel = keras.models.load_model("./saved_models/garnet_rev_37_full.h5", compile=False)

# build new model
model = generateModel((image_shape[0], image_shape[1], 1),
                      output_filters=5,
                      initial_filters=8,
                      depth=8
                      )

with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())

    # load weights from previous run
    model.load_weights("./saved_models/garnet_rev_39_weights.h5", by_name=True)
    # copyWeights(sess, oldModel, model, 'repeatedLogic_', 'repeatedLogic_')

    # link repeated layers
    # linkAllWeights(model)

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
              epochs=10,
              steps_per_epoch=50,
              callbacks=[tensorboard]
              )

    # unlink layers prior to saving
    # unlinkAllWeights(model, sess)

    # save the model
    model.save_weights('./saved_models/garnet_rev_40_weights.h5', save_format='h5')
    model.save("./saved_models/garnet_rev_40_full.h5")

    # dump output before weights are unlinked
    dump_images(sess, model, allDataAndLabels, 100, 0.3,
                channels=label_channels)

    # save output samples and exit
    exit()
