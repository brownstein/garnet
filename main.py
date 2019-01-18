import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights, unlinkWeights, copyWeights
# from data.small_shapes_2 import load_dataset
from data.small_combinations import load_dataset
from run_output import dump_images_2
from loss import loss

allDataAndLabels = load_dataset(dtype=tf.float16,
                                input_shape=(40, 40),
                                label_shape=(40, 40))

# load old model first so that the new one can initialize properly
# oldModel = keras.models.load_model("garnet_r15.h5", compile=False)
# oldModel.load_weights("./saved_models/garnet_r15")

# build new model
model = generateModel((40, 40, 2),
                      output_filters=5,
                      initial_filters=8,
                      logic_filters=32,
                      kernel_size=7,
                      rec_depth=30,
                      prefix=''
                      )

with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())

    # model = keras.models.load_model("garnet_r16.h5", compile=False)
    model.load_weights("./saved_models/garnet_r16", by_name=True)

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
    model.fit(allDataAndLabels, epochs=50, steps_per_epoch=50, callbacks=[tensorboard])

    # unlink layers prior to saving
    unlinkWeights(model, sess)

    # save the model
    model.save_weights('./saved_models/garnet_r16', save_format='h5')
    model.save("garnet_r16.h5")

    # save output samples and exit
    dump_images_2(sess, model, allDataAndLabels, 1, 100)
    exit()
