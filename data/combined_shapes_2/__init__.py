import tensorflow as tf
from os import path

NUM_TEST_CASES = 200
DATA_FILENAMES = {
    "edges":  "shapes_edges_{0}.png",
    "filled": "shapes_filled_{0}.png"
}
LABEL_FILENAMES = {
    "edges":           "shapes_edges_{0}.png",
    "filled":          "shapes_filled_{0}.png",
    "edges_circle":    "shapes_edges_circle_{0}.png",
    "edges_square":    "shapes_edges_square_{0}.png",
    "edges_triangle":  "shapes_edges_triangle_{0}.png",
    "filled_circle":   "shapes_filled_circle_{0}.png",
    "filled_square":   "shapes_filled_square_{0}.png",
    "filled_triangle": "shapes_filled_triangle_{0}.png"
}

def load_dataset(dtype=tf.float16,
                 data_shape=(64, 64),
                 label_shape=(64, 64),
                 data_channels=(
                    'edges',
                    'filled'
                 ),
                 label_channels=(
                    'edges',
                    'filled',
                    'filled_circle',
                    'filled_square',
                    'filled_triangle'
                 ),
                 num_cases=NUM_TEST_CASES,
                 interleve_data_cases=True
                 ):

    cd = path.dirname(__file__)

    caseArrays = []
    for n in range(num_cases):
        caseArray = []
        if interleve_data_cases:
            if n % 2:
                caseArray.append(path.join(cd, "data", DATA_FILENAMES["edges"].format(n)))
                caseArray.append(path.join(cd, "data", DATA_FILENAMES["edges"].format(n)))
            else:
                caseArray.append(path.join(cd, "data", DATA_FILENAMES["filled"].format(n)))
                caseArray.append(path.join(cd, "data", DATA_FILENAMES["filled"].format(n)))
        else:
            for channel in data_channels:
                caseArray.append(path.join(cd, "data", DATA_FILENAMES[channel].format(n)))
        for channel in label_channels:
            caseArray.append(path.join(cd, "labels", LABEL_FILENAMES[channel].format(n)))
        caseArrays.append(caseArray)

    def get_tensors(file_list):
        dataLayers = []
        labelLayers = []

        for channelNo in range(len(data_channels)):
            imageString = tf.read_file(file_list[channelNo])
            imageTensor = tf.image.decode_png(imageString, 1)
            imageTensor = tf.cast(imageTensor, dtype)
            imageTensor = tf.image.resize_images(imageTensor, data_shape)
            dataLayers.append(imageTensor)

        for channelNo in range(len(data_channels), len(data_channels) + len(label_channels)):
            imageString = tf.read_file(file_list[channelNo])
            imageTensor = tf.image.decode_png(imageString, 1)
            imageTensor = tf.cast(imageTensor, dtype)
            imageTensor = tf.image.resize_images(imageTensor, label_shape)
            labelLayers.append(imageTensor)

        dataStack = tf.concat(dataLayers, len(dataLayers[0].shape) - 1)
        labelStack = tf.concat(labelLayers, len(labelLayers[0].shape) - 1)

        return (dataStack, labelStack)

    dataset = tf.data.Dataset.from_tensor_slices(caseArrays).map(get_tensors)
    return dataset.shuffle(num_cases).repeat().batch(20)
