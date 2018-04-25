from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from config import *

'''
This file contains the model used for the project
'''

class Model:
    # The inputs passed to the model and the batch size are the class variables used by all the functions
    # inputs are 4D tensors [batch_size, HEIGHT, WIDTH, CHANNELS]
    # batch_size is a scalar values representing the number of passed in the inputs tensor
    # Collecting the mean values used by the model with collect_means
    # Collecting the locations using collect_locs
    def __init__(self, inputs, b_size):
        self.inputs = inputs
        self.batch_size = b_size * NUM_EPISODES
        self.collect_locs = []
        self.collect_means = []
        

    # This function calculated the inital locations, then build the LSTM cell 
    # the output of this function is the classifier output of the last LSTM cell 
    # class_outs is a 2D tensor of dimenstions [batch_size, number_of_claases]
    def __call__(self):
        initial_locs = tf.random_uniform([self.batch_size, LOC_DIM], minval=-1, maxval=1)

        # print(initial_locs)
        
        input_lstm = self.glimpse_network(self.inputs, initial_locs)

        collect_outputs= []
        baselines = []
        prev_output = tf.zeros([self.batch_size, 12, 12, 24])
        prev_state = tf.zeros([self.batch_size, 12, 12, 24])

        curr_out, next_state = self.peephole_lstm(prev_output, prev_state, input_lstm)
        prev_state = next_state
        prev_output = self.next_location(curr_out, False)
        
        for i in range(NUM_GLIMPSES):
            curr_out, next_state = self.peephole_lstm(prev_output, prev_state, input_lstm)
            collect_outputs.append(curr_out)
            base = self.baseline_layer(curr_out, GLIMPSE_FC2, BASE_OUT, 'baseline')
            baselines.append(base)
            prev_output = self.next_location(curr_out, True)
            prev_state = next_state

        wc1 = tf.Variable(tf.truncated_normal([1, 1, 24, 24], stddev=0.1))
        bc1 = tf.Variable(tf.constant(0.1, shape=[24]))

        weights = tf.Variable(tf.truncated_normal([216, 1], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[1]))

        pool1 = self.pool_operations(collect_outputs[-1], 'pl1', [1,2,2,1], [1,2,2,1])
        # print(pool1.shape)

        conv1 = self.conv_layer(pool1, 1,24,24, 'c1')

        pool2 = self.pool_operations(conv1, 'pl2', [1,2,2,1], [1,2,2,1])
        # print(pool2.shape)

        fin = tf.reshape(pool2, [-1, 3*3*24])
            
        class_outs = self.fc_layer(fin, 216, 10, 'softmax', None)

        return baselines, class_outs, self.collect_means, self.collect_locs


    def baseline_layer(self, image, in_size, out_size, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
            # biases = tf.get_variable("biases", [out_size], initializer=tf.contrib.layers.xavier_initializer())
            wc1 = tf.Variable(tf.truncated_normal([1, 1, 24, 24], stddev=0.1))
            bc1 = tf.Variable(tf.constant(0.1, shape=[24]))

            weights = tf.Variable(tf.truncated_normal([216, 1], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[1]))

            pool1 = self.pool_operations(image, 'poolloc1', [1,2,2,1], [1,2,2,1])
            # print(pool1.shape)

            conv1 = self.conv_layer(pool1, 1,24,24, 'conv1')

            pool2 = self.pool_operations(conv1, 'poolloc2', [1,2,2,1], [1,2,2,1])
            # print(pool2.shape)

            fin = tf.reshape(pool2, [-1, 3*3*24])

            y = tf.nn.tanh(tf.add(tf.matmul(fin, weights), biases))
            # y = tf.stop_gradient(y)

            return y

    

    # This function is called by the current LSTM cell to get inputs to the next cell
    # next_inputs are of dimension [batch_size, 256]
    def next_location(self, prev_inputs, is_first):
        with tf.variable_scope('next_loc', reuse=tf.AUTO_REUSE):
            # weights = tf.get_variable("weights", [GLIMPSE_FC2, LOC_DIM], initializer=tf.contrib.layers.xavier_initializer())
            # biases = tf.get_variable("biases", [LOC_DIM], initializer=tf.contrib.layers.xavier_initializer())

            wc1 = tf.Variable(tf.truncated_normal([1, 1, 24, 24], stddev=0.1))
            bc1 = tf.Variable(tf.constant(0.1, shape=[24]))

            weights = tf.Variable(tf.truncated_normal([216, 2], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[2]))

            pool1 = self.pool_operations(prev_inputs, 'poolloc1', [1,2,2,1], [1,2,2,1])
            # print(pool1.shape)

            conv1 = self.conv_layer(pool1, 1,24,24, 'conv1')

            pool2 = self.pool_operations(conv1, 'poolloc2', [1,2,2,1], [1,2,2,1])
            # print(pool2.shape)

            fin = tf.reshape(pool2, [-1, 3*3*24])

            # print(fin.shape)


            y = tf.add(tf.matmul(fin, weights), biases)
            # y = tf.stop_gradient(y)
            
            means = tf.nn.tanh(y) + tf.random_normal((self.batch_size, LOC_DIM), STD_VAR)
            # means = tf.stop_gradient(means)

            locs = means + tf.random_normal((self.batch_size, LOC_DIM), STD_VAR)

            # locs = tf.stop_gradient(locs)

            if is_first:
                self.collect_locs.append(locs)
                self.collect_means.append(means)

            next_inputs = self.glimpse_network(self.inputs, locs)
            
            return next_inputs


    # This function has the glimpse network where the locations are processed 
    # output is a 2D tensor of dimension [batch_size, 256]
    def glimpse_network(self, input_img, locations):

        locations_d = tf.expand_dims(locations, 2)
        locations_d = tf.expand_dims(locations_d, 3)
        upsample1 = tf.image.resize_images(locations_d, size=(6,6), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv1_out = self.conv_layer(upsample1, 1,1, 6, 'upsample1')

        upsample2 = tf.image.resize_images(conv1_out, size=(12,12), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        conv2_out = self.conv_layer(upsample2, 1,6, 24, 'upsample2')

        glimpses = tf.image.extract_glimpse(input_img, [G_WIN_SIZE,G_WIN_SIZE], 
                                                locations, centered=True, normalized=True)

        g = self.inception(glimpses ,'inception1')
        
        return tf.nn.tanh(g + conv2_out)


    def inception(self, g1, name):
        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            convg1a = self.conv_layer(g1, 1,3,3, 'convg1a')
            convg1b = self.conv_layer(g1, 1,3,3, 'convg1b')
            convg1c = self.pool_operations(g1, 'pool1', [1,2,2,1], [1,1,1,1])

            outg1a = self.conv_layer(g1, 1,3,6, 'outg1a')
            outg1b = self.conv_layer(convg1a, 1,3,6, 'outg1b')
            outg1c = self.conv_layer(convg1b, 3,3,6, 'outg1c')
            outg1d = self.conv_layer(convg1c, 1,3,6, 'outg1d')

            out1 = tf.concat([outg1a, outg1b, outg1c, outg1d], 3)

            # g2 = self.pool_operations(g2, 'pool1', [1,2,2,1], [1,2,2,1])
            # convg2a = self.conv_layer(g2, 1,3,4, 'convg2a')
            # convg2b = self.conv_layer(g2, 1,3,4, 'convg2b')
            # convg2bb = self.conv_layer(convg2b, 3,4,4, 'convg2bb')

            # g3 = self.pool_operations(g3, 'pool2', [1,2,2,1], [1,4,4,1])
            # print(g3.shape)
            # convg3a = self.conv_layer(g3, 1,3,4, 'convg3a')
            # convg3b = self.conv_layer(g3, 1,3,4, 'convg3b')
            # convg3bb = self.conv_layer(convg3b, 3,4,4, 'convg3bb')

            # res = tf.concat([convg1a, convg1bb, convg2a, convg2bb, convg3a, convg3bb], 3)
            # res = tf.concat([convg1a, convg1bb], 3)

            return out1


    def conv_layer(self, in_image, fil_size, no_in, no_out, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights", [fil_size, fil_size, no_in, no_out], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases", [no_out], initializer=tf.contrib.layers.xavier_initializer())

            conv = tf.nn.conv2d(in_image, weights, strides=[1,1,1,1], padding='SAME')
            
            return tf.nn.tanh(conv + biases)

    def pool_operations(self, image1, given_name, ksize_value, stride_value):
        with  tf.variable_scope(given_name, reuse=tf.AUTO_REUSE):
            pool_1 = tf.nn.max_pool(image1, ksize=ksize_value, strides=stride_value, padding='SAME')
            
            return pool_1


    # general template for a fully connected layer used by the model
    # output dimensions are [batch_size, out_size]
    def fc_layer(self, image, in_size, out_size, name, activation):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases", [out_size], initializer=tf.contrib.layers.xavier_initializer())
            y = tf.add(tf.matmul(image, weights), biases)
            # tf.summary.histogram('weights_hist', weights)
            # tf.summary.histogram('biases_hist', biases)

            if activation is not None:
                y = activation(y)

            return y
    
    
    def peephole_lstm(self, last_output, last_state, curr_input):
        
        with tf.variable_scope('lstm_cell', reuse=tf.AUTO_REUSE):

            wxi = tf.get_variable("wxi", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            whi = tf.get_variable("whi", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            wci = tf.get_variable("wci", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            bi = tf.get_variable("bi", [24], initializer=tf.contrib.layers.xavier_initializer())

            wxf = tf.get_variable("wxf", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            whf = tf.get_variable("whf", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            wcf = tf.get_variable("wcf", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            bf = tf.get_variable("bf", [24], initializer=tf.contrib.layers.xavier_initializer())

            wxc = tf.get_variable("wxc", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            whc = tf.get_variable("whc", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            bc = tf.get_variable("bc", [24], initializer=tf.contrib.layers.xavier_initializer())

            wxo = tf.get_variable("wxo", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            who = tf.get_variable("who", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            wco = tf.get_variable("wco", [1, 1, 24, 24], initializer=tf.contrib.layers.xavier_initializer())
            bo = tf.get_variable("bo", [24], initializer=tf.contrib.layers.xavier_initializer())

            it = tf.nn.sigmoid(tf.add((tf.nn.conv2d( curr_input, wxi, strides=[1,1,1,1], padding='SAME') + 
                tf.nn.conv2d(last_output, whi, strides=[1,1,1,1], padding='SAME') +
                tf.nn.conv2d(last_state, wci, strides=[1,1,1,1], padding='SAME') ), bi) ) 

            ft = tf.nn.sigmoid(tf.add((tf.nn.conv2d(curr_input, wxf, strides=[1,1,1,1], padding='SAME') + 
                tf.nn.conv2d(last_output, whf, strides=[1,1,1,1], padding='SAME') +
                tf.nn.conv2d(last_state, wcf, strides=[1,1,1,1], padding='SAME') ), bf) ) 


            ct_mix = tf.nn.tanh(tf.add((tf.nn.conv2d(curr_input, wxc, strides=[1,1,1,1], padding='SAME') + 
                tf.nn.conv2d(last_output, whc, strides=[1,1,1,1], padding='SAME') ), bc) ) 
            ct = tf.multiply(ft, last_state) + tf.multiply(it, ct_mix)


            ot = tf.nn.sigmoid(tf.add((tf.nn.conv2d(curr_input,wxo, strides=[1,1,1,1], padding='SAME') + 
                tf.nn.conv2d(last_output, who, strides=[1,1,1,1], padding='SAME') +
                tf.nn.conv2d( ct, wco,  strides=[1,1,1,1], padding='SAME') ), bo) ) 

            ht = tf.multiply(ot, tf.nn.tanh(ct))


            return ht, ct


            
            
        
        
