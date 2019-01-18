import tensorflow as tf
from os import listdir, path


# gets a nested dict mapping test case number -> channel -> filename
def enumerateDirectoryAsNestedDict(dir):
    testNumberToAttributes = {}
    for filename in listdir(dir):
        if not path.isfile(path.join(dir, filename)):
            continue
        (name, ext) = path.basename(filename).split('.')
        if ext != 'png':
            continue
        nameParts = name.split('-')
        if len(nameParts) < 2:
            continue
        attrName = '-'.join(nameParts[0:-1])
        testNumber = nameParts[-1]
        if testNumber not in testNumberToAttributes:
            testNumberToAttributes[testNumber] = {}
        testNumberToAttributes[testNumber][attrName] = path.join(dir, filename)
    return testNumberToAttributes

# gets a sequence of valid test numbers and dicts to index into them
def enumerateValidTestCases(dir, data_channels, label_channels, require_labels=False):
    dataDict = enumerateDirectoryAsNestedDict(path.join(dir, "data"))
    labelsDict = enumerateDirectoryAsNestedDict(path.join(dir, "labels"))
    testNumbers = []

    for testNo in dataDict.keys():
        if testNo not in labelsDict:
            continue
        allValid = True
        for channel in data_channels:
            if channel not in dataDict[testNo]:
                allValid = False
        for channel in label_channels:
            if require_labels and channel not in labelsDict[testNo]:
                allValid = False
        if allValid:
            testNumbers.append(testNo)
    return (testNumbers, dataDict, labelsDict)


# creates a dataset for a given subdirectory
def createImageDataSetForDirectory(dir,
                                   dtype=tf.float16,
                                   data_channels=(
                                        'fill',
                                        'edges'
                                    ),
                                   label_channels=(
                                        # input channel passthroughs... for now
                                        'fill',
                                        'edges',
                                        # rgb-shapes
                                        'triangularity',
                                        'squareness',
                                        'circularity'
                                   ),
                                   data_shape=(40, 40),
                                   label_shape=(40, 40)
                                   ):
    # make sure the order matches here; using some jank logic
    rgb_expands_to=('triangularity', 'squareness', 'circularity')
    (caseNumbers, dataDict, labelsDict) = enumerateValidTestCases(
        dir,
        data_channels,
        label_channels=(
            'fill',
            'edges',
            'rgb-shapes'
        )
    )

    caseArrays = []
    for n in caseNumbers:
        caseArray = []
        for channel in data_channels:
            caseArray.append(dataDict[n][channel])
            caseArray.append(dataDict[n][channel])
            for channel in label_channels:
                if channel in labelsDict[n]:
                    caseArray.append(labelsDict[n][channel])
            for channel in rgb_expands_to:
                if 'rgb-shapes' in labelsDict[n]:
                    caseArray.append(labelsDict[n]['rgb-shapes'])
            caseArrays.append(caseArray)

    def getTensors(file_list):
        dataLayers = []
        labelLayers = []

        # handle input channelsand labels
        for channelNo in range(len(data_channels)):
            imageString = tf.read_file(file_list[channelNo])
            imageTensor = tf.image.decode_png(imageString, 1)
            imageTensor = tf.cast(imageTensor, dtype)
            imageTensor = tf.image.resize_images(imageTensor, data_shape)
            dataLayers.append(imageTensor)
        for channelNo in range(len(data_channels), len(data_channels) + len(label_channels) - 3):
            imageString = tf.read_file(file_list[channelNo])
            imageTensor = tf.image.decode_png(imageString, 1)
            imageTensor = tf.cast(imageTensor, dtype)
            imageTensor = tf.image.resize_images(imageTensor, label_shape)
            labelLayers.append(imageTensor)

        # handle RGB attributes
        rgbChannelNo = len(data_channels) + len(label_channels) - 3
        imageString = tf.read_file(file_list[rgbChannelNo])
        imageTensor = tf.image.decode_png(imageString, 3)
        imageTensor = tf.cast(imageTensor, dtype)
        imageTensor = tf.image.resize_images(imageTensor, label_shape)
        labelLayers.append(imageTensor)

        dataStack = tf.concat(dataLayers, len(dataLayers[0].shape) - 1)
        labelStack = tf.concat(labelLayers, len(labelLayers[0].shape) - 1)

        return (dataStack, labelStack)

    return tf.data.Dataset.from_tensor_slices(caseArrays).map(getTensors)

# loads the dataset
def load_dataset(dtype=tf.float16,
                 input_shape=(40, 40),
                 label_shape=(40, 40)
                 ):
    cd = path.dirname(__file__)

    everything = createImageDataSetForDirectory(
        cd,
        data_shape=input_shape,
        label_shape=label_shape
        )

    return everything.shuffle(50).repeat().batch(20)
