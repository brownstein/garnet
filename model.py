import tensorflow as tf
from tensorflow import keras

# Linked layer stack helper - this is how you build RCNNs cheaply!
def LinkedConv2DStack (rec_depth=4, kernel_size=5, logic_filters=32,
    initial_filters=16, output_filters=8, activation='selu', padding='same',
    data_format='channels_last', prefix=''):
    def apply (inputLayer, inputLogits=None):

        # this maps logit suggestions to initial logic weights, or the input
        # layer if no logit suggestion layer is specified
        initialLogitLayer = keras.layers.Conv2D(
            filters=initial_filters,
            kernel_size=kernel_size,
            padding=padding,
            data_format=data_format,
            activation=activation,
            name="{0}conv2d".format(prefix)
            )

        concatAxis = 3 if data_format == 'channels_last' else 1
        chain = keras.layers.Concatenate(concatAxis)([
            inputLayer,
            initialLogitLayer(
                inputLayer
                if inputLogits is None
                else inputLogits
                )
            ])

        initialLogitChain = chain

        # this is where the good stuff happens - concept diffusion and
        # intersection processing by way of filter mixing.
        # trained at variable depths, you can force partial recursive
        # invariance, which should do a good job of ensuring the filters
        # become specialized
        for r in range(0, rec_depth):
            chain = keras.layers.SeparableConv2D(
                filters=logic_filters,
                kernel_size=kernel_size,
                padding=padding,
                data_format=data_format,
                activation=activation,
                name="{0}repeatedConv2D_{1}".format(prefix, r)
                )(chain)
            chain = keras.layers.Concatenate(concatAxis)([
                initialLogitChain,
                chain
                ])

        # remap down onto fewer channels
        chain = keras.layers.SeparableConv2D(
            filters = logic_filters, kernel_size = 3,
            padding = padding, data_format = data_format,
            activation = activation, name="{0}outputConv2d".format(prefix)
            )(chain)

        # remap output again onto final channels
        outputLogitLayer = keras.layers.SeparableConv2D(
            filters = output_filters, kernel_size = 1,
            padding = padding, data_format = data_format,
            activation = activation, name="{0}outputLogits".format(prefix)
            )(chain)

        return outputLogitLayer
    return apply

# Generates a training-ready model
def generateModel (input_shape=(64, 64, 1), noise_level=0.1, prefix='',
                  **kwargs):

    inputLayer = keras.layers.Input(input_shape)
    chain = keras.layers.GaussianNoise(
        noise_level,
        name="{0}gaussianNoise".format(prefix)
        )(inputLayer)
    chain = LinkedConv2DStack(prefix=prefix, **kwargs)(chain)
    outputLayer = chain

    return keras.Model(inputs=inputLayer, outputs=outputLayer)

# links weights between conv layers
def linkWeights (model, offset=0):
    firstLayer = None
    layerNo = 0
    for layer in model.layers:
        if 'repeatedConv2D_' not in layer.name:
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

# unlinks weights prior to saving
def unlinkWeights (model, sess):
    foundWeights = []
    ops = []
    for layer in model.layers:
        if 'repeatedConv2D_' not in layer.name:
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

# copies weights from one model to another
def copyWeights (sess, fromModel, toModel, fromPrefix='', toPrefix=''):
    successCount = 0
    fromByName = {}
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
           sess.run(tf.assign(layer.bias, fromLayer.bias))
        if hasattr(layer, '_trainable_weights'):
            for i in range(len(layer._trainable_weights)):
                w = layer._trainable_weights[i]
                fromW = fromLayer._trainable_weights[i]
                sess.run(tf.assign(w, fromW))
        if hasattr(layer, 'weights'):
            for i in range(len(layer.weights)):
                w = layer.weights[i]
                fromW = fromLayer.weights[i]
                sess.run(tf.assign(w, fromW))
    return successCount
