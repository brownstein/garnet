from tensorflow import reduce_sum, selu, subtract

def loss(truth, prediction):


    return truth.selu().sub(prediction.selu())
