import tensorflow as tf
from tensorflow import keras
from model import generateModel, linkWeights, copyWeights
# from data.small_shapes import load_data_as_dataset
from data.small_shapes_2 import load_dataset
from run_output import run_output_summaries
from loss import loss

allDataAndLabels = load_dataset(dtype=tf.float16,
                                input_shape=(32, 32),
                                label_shape=(32, 32))

model = generateModel((32, 32, 2),
                      output_filters=6,
                      initial_filters=25,
                      logic_filters=25,
                      kernel_size=7,
                      rec_depth=25)

linkWeights(model)

with tf.Session().as_default() as sess:

    # I had this working in JS... might have to change loss
    # or layers. At least it runs!

    # oldModel = keras.models.load_model("garnet2.h5", compile=False)
    # oldModel.load_weights("garnet2.h5")
    # copyWeights(oldModel, model)

    # model.load_weights("./saved_models/garnet-r8")

    model.summary()
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir="./graph",
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    model.fit(allDataAndLabels, epochs=400, steps_per_epoch=25, callbacks=[tensorboard])

    model.save_weights('./saved_models/garnet-r11')
    model.save("garnet_r11.h5")

    run_output_summaries(sess, model, allDataAndLabels, 0.5, 100)

    exit()
