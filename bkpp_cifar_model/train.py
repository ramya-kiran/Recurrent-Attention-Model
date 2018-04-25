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
from util import *
from tensorflow.python.framework import ops
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(batch_size, epochs, log, output):
    # Read arguments
    filename_queue = tf.train.string_input_producer(['cifar_train.tfrecords'])
    image, label = read_and_decode(filename_queue)
    batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=500, num_threads=2, min_after_dequeue=250)

    # placeholders for the input and labels
    X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH], name='X')
    y = tf.placeholder(tf.float32, [None, 10], name='labels')

    # Model instantiated and called for processing the inputs
    model = Model(X, batch_size)
    b_t, y_hat, means, locs = model()

    class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y,
        logits=y_hat
    ))

    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    # correct_prediction = tf.equal(y_hat, y)
    reward_last_step = tf.expand_dims(tf.cast(correct_prediction, tf.float32), 1)
    rewards = tf.tile(reward_last_step, (1, NUM_GLIMPSES)) 

    log_likelihood = calc_likelihood(means, locs, STD_VAR)
    penalty = rewards - b_t

    del_j = tf.reduce_mean(log_likelihood * penalty)

    baseline_loss = tf.reduce_mean(tf.square((rewards - b_t)))

    loss = (-del_j) + class_loss + baseline_loss

    var_list = tf.trainable_variables()


    grads = tf.gradients(loss, var_list)
    grads, _ = tf.clip_by_global_norm(grads, 5)

    global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    learning_rate = tf.train.exponential_decay(
    1e-03,
    global_step,
    55000//batch_size,
    0.97,
    staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-4)

    opt = tf.train.AdamOptimizer(1e-3)
    train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)


    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.33
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # training the model for a predetermined number of epochs
        for i in range(epochs):
            batch_x, batch_lbl = sess.run(batch)
            batch_x = np.tile(batch_x, [NUM_EPISODES, 1, 1, 1])
            batch_lbl = np.tile(batch_lbl, [NUM_EPISODES, 1])
            sess.run(train_op, feed_dict={X: batch_x, y: batch_lbl})
            acc = sess.run(accuracy, feed_dict={X: batch_x, y: batch_lbl})


            s = sess.run(merged_summary, feed_dict={X: batch_x, y: batch_lbl})
            writer.add_summary(s, i)

            if (i+1) % 275 == 0:
                print('Step {}: {}'.format(i+1, acc))

            if (((i+1) % 275 == 0) and (acc > 0.99)):
                # Please change the directory here to save the model in a different location
                params = saver.save(sess, '/N/u/ramyarao/project/model/{}_{}.ckpt'.format(output, i+1))
                print('Model saved: {}'.format(params))

        coord.request_stop()
        coord.join(threads)
    return
        

if __name__ == '__main__':
    train(200, 275*10000, 'logs', 'model')




