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
        #
        # repeatedLayer = keras.layers.SeparableConv2D(filters=logic_filters,
        #     kernel_size=kernel_size, padding=padding, data_format=data_format,
        #     activation=activation)

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

        # remap output into specced number and position of channels
        outputLogitLayer = keras.layers.SeparableConv2D(
            filters = output_filters, kernel_size = 1,
            padding = padding, data_format = data_format,
            activation = activation)

        return outputLogitLayer(chain)
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
        layer.bias = firstLayer.bias
        layer.__trainable_weights = layer._trainable_weights
        layer.__weights = layer.weights
        layer._trainable_weights = []
        layer._trainable_weights.append(firstLayer.bias)
        for w in range(len(layer.weights)):
            layer.weights[w] = firstLayer.weights[w]
            layer._trainable_weights.append(firstLayer.weights[w])

# unlinks weights prior to saving
def unlinkWeights (model):
    foundWeights = []
    for layer in model.layers:
        if not layer.weights:
            continue
        if hasattr(layer, '__trainable_weights'):
            layer._trainable_weights = layer.__trainable_weights
            del layer.__trainable_weights
        for w in range(len(layer.weights)):
            weight = layer.weights[w]
            if weight in foundWeights:
                layer.weights[w] = tf.Variable(weight)
            foundWeights.append(weight)

# copies weights from one model to another
def copyWeights (fromModel, toModel, fromPrefix='', toPrefix=''):
    fromByName = {}
    for layer in fromModel.layers:
        fromByName[layer.name] = layer
    for layer in toModel.layers:
        fromLayerName = layer.name.replace(toPrefix, fromPrefix, 1)
        if not hasattr(fromByName, fromLayerName):
            return
        fromLayer = fromByName[fromLayerName]
        if not fromLayer:
            print ("unable to find layer {0}".format(fromLayer))
            continue
        # if hasattr(layer, 'bias'):
        #    layer.bias.assign(fromLayer.bias)
        # for i in range(len(layer._trainable_weights)):
        #     w = layer._trainable_weights[i]
        #     fromW = fromLayer._trainable_weights[i]
        #     w.assign(fromW)
        # for i in range(len(layer.weights)):
        #     w = layer.weights[i]
        #     fromW = fromLayer.weights[i]
        #     w.assign(fromW)
