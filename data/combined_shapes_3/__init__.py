import tensorflow as tf
from os import path

NUM_TEST_CASES = 100
DATA_FILENAMES = {
    "edges": "case_{0}_rgb_triangles_squares_circles_edges.png",
    "fill": "case_{0}_rgb_triangles_squares_circles_filled.png",
    "rgb": "case_{0}_rgb_triangles_squares_circles_filled.png"
}
LABEL_FILENAMES = DATA_FILENAMES
RGB_CHANNELS = (
    'triangle',
    'square',
    'circle'
)

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
            if channel in ['edges', 'fill']:
                caseArray.append(path.join(dataPath, LABEL_FILENAMES[channel].format(n)))
            else:
                if channel in RGB_CHANNELS:
                    caseArray.append(path.join(dataPath, LABEL_FILENAMES["rgb"].format(n)))
                else:
                    print("wtf")
                    print(channel)

        caseArrays.append(caseArray)

    def get_tensors(file_list):
        dataLayers = []
        labelLayers = []

        for channelNo in range(len(data_channels)):
            imageString = tf.read_file(file_list[channelNo])
            imageTensor = tf.image.decode_png(imageString, 1)
            imageTensor = tf.cast(imageTensor, dtype)
            imageTensor = tf.clip_by_value(tf.multiply(imageTensor, 16), 0, 1)
            imageTensor = tf.cast(imageTensor, dtype)
            imageTensor = tf.image.resize_images(imageTensor, data_shape)
            dataLayers.append(imageTensor)

        for channelNo in range(len(label_channels)):

            channel = label_channels[channelNo]
            channelNoWithOffset = channelNo + len(data_channels)
            imageString = tf.read_file(file_list[channelNoWithOffset])

            imageTensor = None
            if (channel in ['edges', 'fill']):
                imageTensor = tf.image.decode_png(imageString, 1)
                imageTensor = tf.cast(imageTensor, dtype)
                imageTensor = tf.clip_by_value(tf.multiply(imageTensor, 16), 0, 255)
            else:
                if channel in RGB_CHANNELS:
                    imageTensor = tf.image.decode_png(imageString, 3)
                    channelIndex = RGB_CHANNELS.index(channel)
                    imageTensor = imageTensor[:, :, channelIndex:channelIndex+1]
                    imageTensor = tf.cast(imageTensor, dtype)
                else:
                    print("wtf")
                    return None

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
