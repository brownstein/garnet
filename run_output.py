import tensorflow as tf
import os

def run_output_summaries(sess, model, dataSet):
    iterator = dataSet.make_one_shot_iterator()
    step = 0
    nextCaseAndLabel = iterator.get_next()
    while (nextCaseAndLabel):
        step += 1
        nextCase = nextCaseAndLabel[0]
        p = model.predict(nextCase, steps=1)
        channels = (
            'fill',
            'edges',
            'symmetry',
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
            nextCaseAndLabel = iterator.get_next()
