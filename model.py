import tensorflow as tf
from tensorflow import keras
import math

# helper for linked conv stacks with dilation components
def LinkedConv2DMultiStack (
    # basic props
    depth=10,
    kernel_size=5,
    transfer_dilation=2,

    # filter depths
    initial_filters=8,
    logic_filters=30,
    transfer_filters=4,
    mergedown_filters=36,

    # things that shouldn't change often
    activation='selu',
    padding='same',
    data_format='channels_last',

    # layer prefixes
    initial_logits_prefix='initialLogits_',
    logic_prefix='repeatedLogic_',
    transfer_prefix='repeatedTransfer_',
    mergedown_prefix='repeatedMergeDown_',
    ):
    def apply (input_layer):

        # this maps (and sanitizes) the input layer
        initialLogitLayer = keras.layers.Conv2D(
            filters=initial_filters,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
            activation=activation,
            name="{0}conv2D".format(initial_logits_prefix)
            )

        chain = initialLogitLayer(input_layer)
        initialLogits = chain

        concatAxis = 3 if data_format == 'channels_last' else 1

        # repeat the same linked weights for most of the model's depth
        for d in range(0, depth):

            # create both forks of the chain
            # one is raw conv, one is dialated conv to move data around faster
            logicFork = keras.layers.SeparableConv2D(
                filters=logic_filters,
                kernel_size=kernel_size,
                padding=padding,
                data_format=data_format,
                activation=activation,
                name="{0}conv2D_{1}".format(logic_prefix, d)
            )(chain)

            transferFork = keras.layers.SeparableConv2D(
                filters=transfer_filters,
                kernel_size=kernel_size,
                padding=padding,
                data_format=data_format,
                activation=activation,
                name="{0}conv2D_{1}".format(transfer_prefix, d),
                dilation_rate=transfer_dilation
            )(chain)

            # merge them
            chain = keras.layers.Concatenate(concatAxis)([
                initialLogits,
                logicFork,
                transferFork
            ])

            # decrease the number of channels
            chain = keras.layers.SeparableConv2D(
                filters=mergedown_filters,
                kernel_size=1,
                data_format=data_format,
                activation=activation,
                name="{0}Conv2D_{1}".format(mergedown_prefix, d)
            )(chain)

        return chain
    return apply


# Generates a training-ready model
def generateModel (input_shape=(64, 64, 1),
                   noise_level=0.15,
                   depth=10,
                   output_filters = 8,
                   padding='same',
                   activation='selu',
                   data_format='channels_last',
                   **kwargs
                  ):

    # create input and give it some noise to process
    inputLayer = keras.layers.Input(input_shape)
    chain = inputLayer

    # add noise between linked sets to keep things interesting
    chain = keras.layers.GaussianNoise(
        noise_level,
        name="gaussianNoise_0"
        )(chain)

    # build repeater stack
    chain = LinkedConv2DMultiStack(
        depth=depth,
        data_format=data_format,
        activation=activation,
        padding=padding,
        **kwargs
        )(chain)

    # remap output again onto final channels
    chain = keras.layers.SeparableConv2D(
        filters = output_filters,
        kernel_size = 1,
        padding = padding,
        data_format = data_format,
        activation = activation,
        name = "outputLogits"
        )(chain)

    # construct model and we're done!
    outputLayer = chain
    return keras.Model(inputs=inputLayer, outputs=outputLayer)

# helper to link all weights for a given model
def linkAllWeights (model):
    linkWeights(model, 2, "repeatedLogic_")
    linkWeights(model, 2, "repeatedTransfer_")
    linkWeights(model, 2, "repeatedMergeDown_")

# links weights between conv layers
def linkWeights (model, offset=0, targetLayersWithPrefix='repeatedConv2D_'):
    firstLayer = None
    layerNo = 0
    for layer in model.layers:
        if not layer.name.startswith(targetLayersWithPrefix):
            continue
        layerNo += 1
        if layerNo <= offset:
            continue
        if not firstLayer:
            firstLayer = layer
            continue
        layer._pre_link = {
            '_trainable_weights': layer._trainable_weights,
            'weights': [w for w in layer.weights],
            'bias': layer.bias
        }
        layer.bias = firstLayer.bias
        layer._trainable_weights = []
        for w in range(len(layer.weights)):
            weight = tf.identity(firstLayer.weights[w])
            layer.weights[w] = weight
            layer._trainable_weights.append(weight)

def unlinkAllWeights (model, sess):
    unlinkWeights(model, sess, 'repeatedLogic_')
    unlinkWeights(model, sess, 'repeatedTransfer_')
    unlinkWeights(model, sess, 'repeatedMergeDown_')

# unlinks weights prior to saving
def unlinkWeights (model, sess, targetLayersWithPrefix='repeatedConv2D_'):
    foundWeights = []
    ops = []
    for layer in model.layers:
        if not layer.name.startswith(targetLayersWithPrefix):
            continue
        if not hasattr(layer, "_pre_link"):
            continue
        print("unlinking {0}...".format(layer.name))
        layer._trainable_weights = layer._pre_link['_trainable_weights']
        for w in range(len(layer.weights)):
            linkedWeight = layer.weights[w]
            unlinkedWeight = layer._pre_link['weights'][w]
            ops.append(tf.assign(unlinkedWeight, linkedWeight))
            layer.weights[w] = unlinkedWeight
            layer._trainable_weights[w] = unlinkedWeight
        del layer._pre_link
    sess.run(ops)

# helper to copy weights from one tensor to another
def copySingleTensor(fromTensor, toTensor):
    toShape = [a for a in toTensor.shape]
    fromShape = [a for a in fromTensor.shape]
    for paramNo in range(len(fromShape)):
        if toShape[paramNo] < fromShape[paramNo]:
            fromShape[paramNo] = toShape[paramNo]
            fromTensor = tf.slice(fromTensor, [0 for a in fromShape], fromShape)
        if toShape[paramNo] > fromShape[paramNo]:
            deltas = [[0, 0] for a in fromShape]
            toShapeLen = toShape[paramNo]
            fromShapeLen = fromShape[paramNo]
            if hasattr(toShapeLen, "value"):
                toShapeLen = toShapeLen.value
                fromShapeLen = fromShapeLen.value
            deltas[paramNo] = [
                math.floor((toShapeLen - fromShapeLen) / 2),
                math.ceil((toShapeLen - fromShapeLen) / 2)
            ]
            fromTensor = tf.pad(fromTensor, tf.constant(deltas))
            fromShape[paramNo] = toShape[paramNo]
    return tf.assign(toTensor, fromTensor)

# copies weights from one model to another
def copyWeights (sess, fromModel, toModel, fromPrefix='', toPrefix=''):
    successCount = 0
    fromByName = {}
    ops = []
    print("copying weights from older model...")
    for layer in fromModel.layers:
        fromByName[layer.name] = layer
        print(" | found source layer {0}".format(layer.name))
    for layer in toModel.layers:
        fromLayerName = layer.name.replace(toPrefix, fromPrefix, 1)
        if fromLayerName not in fromByName:
            print ("(!) unable to find layer {0}".format(fromLayerName))
            continue
        fromLayer = fromByName[fromLayerName]
        successCount += 1
        if hasattr(layer, 'bias'):
           ops.append(copySingleTensor(fromLayer.bias, layer.bias))
           #ops.append(tf.assign(layer.bias, fromLayer.bias))
        if hasattr(layer, '_trainable_weights'):
            for i in range(len(layer._trainable_weights)):
                w = layer._trainable_weights[i]
                fromW = fromLayer._trainable_weights[i]
                ops.append(copySingleTensor(fromW, w))
                #ops.append(tf.assign(w, fromW))
        if hasattr(layer, 'weights'):
            for i in range(len(layer.weights)):
                w = layer.weights[i]
                fromW = fromLayer.weights[i]
                ops.append(copySingleTensor(fromW, w))
                #ops.append(tf.assign(w, fromW))
    sess.run(ops)
    return successCount
