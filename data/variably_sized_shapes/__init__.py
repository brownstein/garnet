import tensorflow as tf
from os import path

NUM_TEST_CASES = 200
DATA_FILENAMES = {
    "edges":  "case_{0}_edges.png",
    "fill": "case_{0}_filled.png"
}
LABEL_FILENAMES = {
    "normal": {
        "edges": {
            "triangle": "case_{0}_{1}_edges.png",
            "square": "case_{0}_{1}_edges.png",
            "circle": "case_{0}_{1}_edges.png"
        },
        "fill": {
            "triangle": "case_{0}_{1}_filled.png",
            "square": "case_{0}_{1}_filled.png",
            "circle": "case_{0}_{1}_filled.png"
        }
    },
    "blurred": {
        "edges": {
            "triangle": "case_{0}_{1}_edges_blurred_.png",
            "square": "case_{0}_{1}_edges_blurred_.png",
            "circle": "case_{0}_{1}_edges_blurred_.png"
        },
        "fill": {
            "triangle": "case_{0}_{1}_filled_blurred_.png",
            "square": "case_{0}_{1}_filled_blurred_.png",
            "circle": "case_{0}_{1}_filled_blurred_.png"
        }
    },
}
LABEL_FALLBACK = "fallback.png"

def load_dataset(dtype=tf.float16,
                 data_shape=(64, 64),
                 label_shape=(64, 64),
                 data_channels=(
                    'fill',
                 ),
                 label_channels=(
                    'fill',
                    'edges',
                    'circle',
                    'square',
                    'triangle'
                 ),
                 num_cases=NUM_TEST_CASES,
                 repeat=True
                 ):

    cd = path.dirname(__file__)
    dataPath = path.join(cd, "data")
    labelsPath = path.join(cd, "labels")
    fallbackPath = path.join(cd, "labels", LABEL_FALLBACK)

    numNormalDataChannels = 0
    caseArrays = []

    for n in range(num_cases):
        caseArray = []
        numNormalDataChannels = 0

        # add data channels
        for channel in data_channels:
            caseArray.append(path.join(dataPath, DATA_FILENAMES[channel].format(n)))

        # add label channels
        for channel in label_channels:
            # edges and fills are provided as data, so use data paths
            if channel in ['edges', 'fill']:
                caseArray.append(path.join(dataPath, DATA_FILENAMES[channel].format(n)))
            # otherwise use shape path logic
            else:
                smallShapePath = path.join(labelsPath, LABEL_FILENAMES["blurred"]["fill"][channel].format(n, channel))
                if (path.exists(smallShapePath)):
                    caseArray.append(smallShapePath)
                else:
                    caseArray.append(fallbackPath)
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

        for channelNo in range(len(label_channels)):
            channelNoWithOffset = channelNo + len(data_channels)
            imageString = tf.read_file(file_list[channelNoWithOffset])
            imageTensor = tf.image.decode_png(imageString, 1)
            imageTensor = tf.cast(imageTensor, dtype)
            imageTensor = tf.image.resize_images(imageTensor, label_shape)
            imageTensor = tf.multiply(imageTensor, 1 / 255)
            labelLayers.append(imageTensor)

        dataStack = tf.concat(dataLayers, len(dataLayers[0].shape) - 1)
        labelStack = tf.concat(labelLayers, len(labelLayers[0].shape) - 1)

        return (dataStack, labelStack)

    dataset = tf.data.Dataset.from_tensor_slices(caseArrays).map(get_tensors)
    if repeat:
        return dataset.shuffle(num_cases).repeat().batch(20)
    else:
        return dataset.shuffle(num_cases).batch(20)
