import tensorflow as tf
import os

def dump_keras_weights(model):
    model.save_weights("./keras_saved")

def dump_images(sess, model, dataSet, max=100, caseRatio=1,
    channels = (
        'edges',
        'fill',
        'circles',
        'squares',
        'triangles'
        )
    ):
    iterator = dataSet.make_one_shot_iterator()
    step = 0
    skipCountdown = 0
    nextCaseAndLabel = iterator.get_next(),
    while (nextCaseAndLabel):
        step += 1

        if (step >= max):
            break

        nextCase = nextCaseAndLabel[0]
        nextCaseAndLabel = iterator.get_next()

        if (caseRatio != 1):
            if (skipCountdown >= 0):
                skipCountdown -= 1
                continue
            else:
                skipCountdown = 1 / caseRatio

        predictCase = nextCase[0]

        if (len(predictCase.shape) == 3):
            predictCase = tf.expand_dims(predictCase, 0)

        p = model.predict(predictCase, steps=1)
        c = 0
        for channelName in channels:
            summaryName="out_{0}_{1}.png".format(step, channelName)
            summaryImage = tf.summary.image(name=summaryName,
                                            tensor=p[:,:,:,c:c+1])

            summary = tf.summary.Summary.FromString(summaryImage.eval())
            png = summary.value[0].image.encoded_image_string
            f = open("output/{0}".format(summaryName), "wb")
            f.write(png)
            f.close()
            c += 1
