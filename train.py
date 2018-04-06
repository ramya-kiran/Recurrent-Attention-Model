'''
Train function
Processes inputs in mini-batchs 
Builds the model and trains the parameters for predetermined number of times 
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os 
from tensorflow.python.framework import ops
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(batch_size, epochs, log, output):
    # Read arguments
    ops.reset_default_graph()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # placeholders for the input and labels
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='X')
    y = tf.placeholder(tf.float32, [None, 10], name='labels')

    y_hat = model(X, batch_size)

    '''
    Calculate loss and then define the Optimizer and related hyper-parameters
    '''
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.33
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # training the model for a predetermined number of epochs
        for i in range(epochs):
            batch = mnist.train.next_batch(batch_size)
            acc = sess.run(y_hat, feed_dict={X: batch[0], y: batch[1]})
            s = sess.run(merged_summary, feed_dict={X: batch[0], y: batch[1]})
            writer.add_summary(s, i)

            if (i+1) % 275 == 0:
                print('Step {}: {:.2f}'.format(i+1, acc))

            if (((i+1) % 275 == 0) and (acc > 0.99)):
                # Please change the directory here to save the model in a different location
                params = saver.save(sess, '/N/u/ramyarao/dl_hw2/exp_model/{}_{}.ckpt'.format(output, i+1))
                print('Model saved: {}'.format(params))

        coord.request_stop()
        coord.join(threads)
    return
        

if __name__ == '__main__':
    train(200, 275*10000, 'logs', 'model')



