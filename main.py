import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights, copyWeights
from data.small_combinations import load_data_as_dataset
from run_output import run_output_summaries
from loss import loss

allDataAndLabels = load_data_as_dataset(dtype=tf.float16)

model = generateModel((64, 64, 2),
                      output_filters=6,
                      initial_filters=6,
                      logic_filters=20,
                      kernel_size=9,
                      rec_depth=32)

linkWeights(model)

with tf.Session().as_default() as sess:

    # I had this working in JS... might have to change loss
    # or layers. At least it runs!

    # oldModel = keras.models.load_model("garnet2.h5", compile=False)
    # oldModel.load_weights("garnet2.h5")
    # copyWeights(oldModel, model)

    # model.load_weights("./saved_models/garnet-r7")

    model.summary()
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir="./graph",
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    model.fit(allDataAndLabels, epochs=500, steps_per_epoch=25, callbacks=[tensorboard])

    model.save_weights('./saved_models/garnet-r8')
    model.save("garnet_r8.h5")

    run_output_summaries(sess, model, allDataAndLabels, 0.1)
