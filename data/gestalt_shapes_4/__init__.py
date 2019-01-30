import tensorflow as tf
from os import path

NUM_TEST_CASES = 100
DATA_FILENAMES = {
    "edges":  "case_{0}_edges.png",
    "fill": "case_{0}_filled.png"
}
LABEL_FILENAMES = {
    "small": {
        "edges": {
            "triangle": "case_{0}_small_triangles_edges.png",
            "square": "case_{0}_small_squares_edges.png",
            "circle": "case_{0}_small_circles_edges.png"
        },
        "fill": {
            "triangle": "case_{0}_small_triangles_filled.png",
            "square": "case_{0}_small_squares_filled.png",
            "circle": "case_{0}_small_circles_filled.png"
        }
    },
    "gestalt": {
        "edges": {
            "triangle": "case_{0}_gestalt_triangle_edges.png",
            "square": "case_{0}_gestalt_square_edges.png",
            "circle": "case_{0}_gestalt_circle_edges.png"
        },
        "fill": {
            "triangle": "case_{0}_gestalt_triangle_filled.png",
            "square": "case_{0}_gestalt_square_filled.png",
            "circle": "case_{0}_gestalt_circle_filled.png"
        }
    }
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
                 include_gestalt_shapes=False,
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
                caseArray.append(fallbackPath)
            # otherwise use shape path logic
            else:
                smallShapePath = path.join(labelsPath, LABEL_FILENAMES["small"]["fill"][channel].format(n))
                gestaltPath = path.join(labelsPath, LABEL_FILENAMES["gestalt"]["fill"][channel].format(n))
                if (path.exists(smallShapePath)):
                    caseArray.append(smallShapePath)
                else:
                    caseArray.append(fallbackPath)
                if (include_gestalt_shapes and path.exists(gestaltPath)):
                    caseArray.append(gestaltPath)
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
            channelNoWithOffset = channelNo * 2 + len(data_channels)
            subImageString = tf.read_file(file_list[channelNoWithOffset])
            subImageTensor = tf.image.decode_png(subImageString, 1)
            subImageTensor = tf.cast(subImageTensor, dtype)
            subImageTensor = tf.image.resize_images(subImageTensor, label_shape)
            subImageTensor = tf.multiply(subImageTensor, 1 / 256)
            gestaltImageString = tf.read_file(file_list[channelNoWithOffset + 1])
            gestaltImageTensor = tf.image.decode_png(gestaltImageString, 1)
            gestaltImageTensor = tf.cast(gestaltImageTensor, dtype)
            gestaltImageTensor = tf.image.resize_images(gestaltImageTensor, label_shape)
            gestaltImageTensor = tf.multiply(gestaltImageTensor, 1 / 256)
            combinedImageTensor = tf.add(subImageTensor, gestaltImageTensor)
            labelLayers.append(combinedImageTensor)

        dataStack = tf.concat(dataLayers, len(dataLayers[0].shape) - 1)
        labelStack = tf.concat(labelLayers, len(labelLayers[0].shape) - 1)

        return (dataStack, labelStack)

    dataset = tf.data.Dataset.from_tensor_slices(caseArrays).map(get_tensors)
    if repeat:
        return dataset.shuffle(num_cases).repeat().batch(20)
    else:
        return dataset.shuffle(num_cases).batch(20)
