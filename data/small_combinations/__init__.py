import tensorflow as tf
from os import listdir, path

def readAttributeImagesFromDir (dir, dtype=tf.uint8):
    idToChannels = {}
    for filename in listdir(dir):
        if not path.isfile(path.join(dir, filename)):
            continue
        (name, ext) = path.basename(filename).split(".")
        if ext != 'png':
            continue
        nameParts = name.split("-")
        if len(nameParts) < 2:
            continue
        attrName = '-'.join(nameParts[0:-1])
        testNumber = nameParts[-1]
        imageString = tf.read_file(path.join(dir, filename))
        if testNumber not in idToChannels:
            idToChannels[testNumber] = {}
        if ('rgb' in nameParts):
            imageTensor = tf.image.decode_png(imageString, 4)
            idToChannels[testNumber][attrName] = tf.cast(imageTensor, dtype)
        else:
            imageTensor = tf.image.decode_png(imageString, 1)
            idToChannels[testNumber][attrName] = tf.cast(imageTensor, dtype)
    return idToChannels

def load_data_as_dataset(
    max_tests=100,
    dtype=tf.uint8,
    input_shape=(64, 64),
    label_shape=(64, 64),
    data_attributes=('fill', 'edges'),
    label_attributes=('fill', 'edges', 'symmetry',
        'circularity', 'squareness', 'triangularity'),
    cd=None
    ):
    cd = cd or (path.dirname(__file__))
    subdirs = filter(path.isdir, map(lambda d: path.join(cd, d), listdir(cd)))

    dataById = readAttributeImagesFromDir(path.join(cd, "data"), dtype=dtype)
    labelsById = readAttributeImagesFromDir(path.join(cd, "labels"), dtype=dtype)

    def combined_generator():
        for id in dataById.keys():
            if id not in labelsById:
                continue
            dataByAttr = dataById[id]
            labelsByAttr = labelsById[id]

            dataStack = []
            for attr in data_attributes:
                if attr in dataByAttr:
                    nextLayer = dataByAttr[attr]
                    dataStack = dataStack + [nextLayer]
                else:
                    dataStack = dataStack + [
                        tf.zeros(shape=(input_shape[0], input_shape[1], 1))
                        ]

            labelStack = []
            for attr in label_attributes:
                if attr in labelsByAttr:
                    nextLayer = labelsByAttr[attr]
                    labelStack = labelStack + [tf.to_float(nextLayer)]
                else:
                    labelStack = labelStack + [
                        tf.zeros(shape=(input_shape[0], input_shape[1], 1))
                        ]

            dataStack = list(tf.image.resize_images(d, input_shape) for d in dataStack)
            dataStack = tf.concat(dataStack, len(dataStack[0].shape) - 1)

            labelStack = list(tf.image.resize_images(l, label_shape) for l in labelStack)
            labelStack = tf.concat(labelStack, len(labelStack[0].shape) - 1)

            if (dtype not in [tf.float16, tf.float32]):
                dataStack = tf.to_int32(dataStack)
                labelStack = tf.to_int32(labelStack)

            yield (dataStack, labelStack)

    tensorPairs = [pair for pair in combined_generator()]
    tensorData = [d for (d, _) in tensorPairs]
    tensorLabels = [l for (_, l) in tensorPairs]

    dSet = tf.data.Dataset.from_tensor_slices((tensorData, tensorLabels))
    return dSet.shuffle(50).repeat().batch(25)
