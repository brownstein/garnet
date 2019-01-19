import tensorflow as tf
import os

def dump_keras_weights(model):
    model.save_weights("./keras_saved")

def dump_images(sess, model, dataSet, outputRatio, max=100,
                channels=(
                    'filled',
                    'edges',
                    'symmetry',
                    'circularity',
                    'squareness',
                    'triangularity'
                )
    ):
    iterator = dataSet.make_one_shot_iterator()
    step = 0
    skipCountdown = 0
    nextCaseAndLabel = iterator.get_next()
    while (nextCaseAndLabel):
        step += 1

        if (step >= max):
            break

        nextCase = nextCaseAndLabel[0]
        nextCaseAndLabel = iterator.get_next()

        if (skipCountdown > 0):
            skipCountdown -= outputRatio
            continue

        skipCountdown = 1

        p = model.predict(nextCase, steps=1)
        c = 0
        for channelName in channels:
            summaryName="out_{0}_{1}.png".format(step, channelName)
            summaryImage = tf.summary.image(name=summaryName,
                                            tensor=p[:,:,:,c:c+1])
            c += 1
            summary = tf.summary.Summary.FromString(summaryImage.eval())
            png = summary.value[0].image.encoded_image_string
            f = open("output/{0}".format(summaryName), "wb")
            f.write(png)
            f.close()

def dump_images_2(sess, model, dataSet, outputRatio, max=100):
    iterator = dataSet.make_one_shot_iterator()
    step = 0
    skipCountdown = 0
    nextCaseAndLabel = iterator.get_next()
    while (nextCaseAndLabel):
        step += 1

        if (step >= max):
            break

        nextCase = nextCaseAndLabel[0]
        nextCaseAndLabel = iterator.get_next()

        if (skipCountdown > 0):
            skipCountdown -= outputRatio
            continue

        skipCountdown = 1

        p = model.predict(nextCase, steps=1)
        channels = (
            'edges',
            'fill',
            'circularity',
            'squareness',
            'triangularity'
        )
        c = 0
        for channelName in channels:
            summaryName="out_{0}_{1}.png".format(step, channelName)
            summaryImage = tf.summary.image(name=summaryName,
                                            tensor=p[:,:,:,c:c+1])
            c += 1
            summary = tf.summary.Summary.FromString(summaryImage.eval())
            png = summary.value[0].image.encoded_image_string
            f = open("output/{0}".format(summaryName), "wb")
            f.write(png)
            f.close()
