import tensorflow as tf
from os import listdir, path

# reads a given directory full of attribute-named images
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

def load_data (
    max_tests=100,
    dtype=tf.uint8,
    input_shape=(64, 64),
    label_shape=(64, 64),
    data_attributes=('fill', 'edges'),
    label_attributes=('fill', 'edges', 'symmetry',
        'circularity', 'squareness', 'triangularity')
    ):
    cd = (path.dirname(__file__))
    subdirs = filter(path.isdir, map(lambda d: path.join(cd, d), listdir(cd)))
    allData = []
    allLabels = []
    for subdir in subdirs:
        dataById = readAttributeImagesFromDir(path.join(subdir, "data"), dtype=dtype)
        labelsById = readAttributeImagesFromDir(path.join(subdir, "labels"), dtype=dtype)
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

            labelStack = list(tf.image.resize_images(l, label_shape) for l in labelStack)
            labelStack = tf.concat(labelStack, len(labelStack[0].shape) - 1)

            for d in dataStack:
                allData = allData + [tf.image.resize_images(d, input_shape)]
                allLabels = allLabels + [labelStack]

    allData = allData[0:max_tests]
    allLabels = allLabels[0:max_tests]

    return (
        tf.stack(allData, 0),
        tf.stack(allLabels, 0)
        )
