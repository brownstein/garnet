import tensorflow as tf
from tensorflow import keras

# load old model first so that the new one can initialize properly
oldModel = keras.models.load_model("garnet_r14A.h5", compile=False)
exit()
