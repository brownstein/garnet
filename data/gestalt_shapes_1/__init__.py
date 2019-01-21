import tensorflow as tf
from os import path

NUM_TEST_CASES = 200
DATA_FILENAMES = {
    "edges":  "case_{0}_edges.png",
    "filled": "case_{0}_filled.png"
}
LABEL_FILENAMES = {
    "edges":            "case_{0}_edges.png",
    "filled":           "case_{0}_filled.png",
    "edges_circles":    "case_{0}_edges_triangles.png",
    "edges_squares":    "case_{0}_edges_squares.png",
    "edges_triangles":  "case_{0}_edges_triangles.png",
    "circles":          "case_{0}_filled_circles.png",
    "squares":          "case_{0}_filled_squares.png",
    "triangles":        "case_{0}_filled_triangles.png",
    "gestalt_circle":   "case_{0}_gestalt_circle.png",
    "gestalt_square":   "case_{0}_gestalt_square.png",
    "gestalt_triangle": "case_{0}_gestalt_triangle.png"
}
LABEL_FALLBACK = "fallback.png"
PLURAL_IMAP = {
    "circles":   "circle",
    "triangles": "triangle",
    "squares":   "square"
}

def load_dataset(dtype=tf.float16,
                 data_shape=(64, 64),
                 label_shape=(64, 64),
                 data_channels=(
                    'edges',
                 ),
                 label_channels=(
                    'filled',
                    'circles',
                    'squares',
                    'triangles'
                 ),
                 num_cases=NUM_TEST_CASES,
                 interleve_data_cases=True
                 ):

    cd = path.dirname(__file__)
    fallback_path = path.join(cd, "labels", LABEL_FALLBACK)

    numNormalDataChannels = 0
    caseArrays = []
    for n in range(num_cases):
        caseArray = []
        numNormalDataChannels = 0
        for channel in data_channels:
            caseArray.append(path.join(cd, "data", DATA_FILENAMES[channel].format(n)))
        for channel in label_channels:
            if channel in ['edges', 'filled']:
                caseArray.append(path.join(cd, "labels", LABEL_FILENAMES[channel].format(n)))
                numNormalDataChannels += 1
            else:
                subshape_path = path.join(cd, "labels", LABEL_FILENAMES[channel].format(n))
                gestalt_channel = "gestalt_{0}".format(PLURAL_IMAP[channel])
                gestalt_path = path.join(cd, "labels", LABEL_FILENAMES[gestalt_channel].format(n))
                if (path.exists(subshape_path)):
                    caseArray.append(subshape_path)
                else:
                    caseArray.append(fallback_path)
                if path.exists(gestalt_path):
                    caseArray.append(gestalt_path)
                else:
                    caseArray.append(fallback_path)
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

        for channelNo in range(1):
            channelNoWithOffset = channelNo + len(data_channels)
            imageString = tf.read_file(file_list[channelNoWithOffset])
            imageTensor = tf.image.decode_png(imageString, 1)
            imageTensor = tf.cast(imageTensor, dtype)
            imageTensor = tf.image.resize_images(imageTensor, data_shape)
            labelLayers.append(tf.multiply(imageTensor, 0.25))

        for channelNo in range(len(label_channels) - numNormalDataChannels):
            channelNoWithOffset = channelNo * 2 + numNormalDataChannels + len(data_channels)
            subImageString = tf.read_file(file_list[channelNoWithOffset])
            subImageTensor = tf.image.decode_png(imageString, 1)
            subImageTensor = tf.cast(imageTensor, dtype)
            subImageTensor = tf.image.resize_images(imageTensor, label_shape)
            gestaltImageString = tf.read_file(file_list[channelNoWithOffset + 1])
            gestaltImageTensor = tf.image.decode_png(imageString, 1)
            gestaltImageTensor = tf.cast(imageTensor, dtype)
            gestaltImageTensor = tf.image.resize_images(imageTensor, label_shape)
            combinedImageTensor = tf.add(tf.multiply(0.5, subImageTensor), gestaltImageTensor)
            labelLayers.append(combinedImageTensor)

        dataStack = tf.concat(dataLayers, len(dataLayers[0].shape) - 1)
        labelStack = tf.concat(labelLayers, len(labelLayers[0].shape) - 1)

        return (dataStack, labelStack)

    dataset = tf.data.Dataset.from_tensor_slices(caseArrays).map(get_tensors)
    return dataset.shuffle(num_cases).repeat().batch(20)
