'''
Train function
Processes inputs in mini-batchs 
Builds the model and trains the parameters for predetermined number of times 
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os 
from config import *
from model import *
from tensorflow.python.framework import ops
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(train_tfrecords, batch_size, epochs, log, output):
    # Read arguments
    # ops.reset_default_graph()
    filename_queue = tf.train.string_input_producer([train_tfrecords])
    image, label = read_and_decode(filename_queue)
    batch = tf.train.shuffle_batch([image, label], batch_size=args.batch_size, capacity=100, num_threads=2, min_after_dequeue=40)

    # placeholders for the input and labels
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='X')
    y = tf.placeholder(tf.float32, [None, 10], name='labels')

    # Model instantiated and called for processing the inputs
    model = Model(X, batch_size)
    lstm_output, y_hat, collect_means, collect_locs = model()

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y,
        logits=y_hat
    ))

    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))

    rewards = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    '''
    Calculate loss and then define the Optimizer and related hyper-parameters
    '''

    b_t = tf.reduce_mean(rewards) # 1 x T tensor
    
    gradient = tf.reduce_mean(tf.reduce_mean(tf.tensordot(tf.multiply(tf.transpose(lstm_output, perm=[2, 0, 1]),
                                                                      tf.subtract(rewards, b_t)), 
                                                          tf.subtract(collect_locs, collect_means),
                                                          axes=[[2], [1]]),
                                             axis=1), 
                              axis=1) # a [K x 2] tensor

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
            img, lbl = sess.run(batch)
            acc = sess.run(y_hat, feed_dict={X: img, y: lbl})
            s = sess.run(merged_summary, feed_dict={X: img, y: lbl})
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
    train('', 200, 275*10000, 'logs', 'model')



