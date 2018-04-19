from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# from config import *
# tf.enable_eager_execution()

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
        self.batch_size = b_size
        self.collect_locs = []
        self.collect_means = []
        

    # This function calculated the inital locations, then build the LSTM cell 
    # the output of this function is the classifier output of the last LSTM cell 
    # class_outs is a 2D tensor of dimenstions [batch_size, number_of_claases]
    def __call__(self):
        initial_locs = tf.random_uniform([self.batch_size, LOC_DIM], minval=-1, maxval=1)
        
        input_lstm = self.glimpse_network(self.inputs, initial_locs)

        collect_outputs= []
        prev_output = tf.zeros([self.batch_size, LSTM_HIDDEN])
        prev_state = tf.zeros([self.batch_size, LSTM_HIDDEN])
        
        for i in range(NUM_GLIMPSES):
            curr_out, next_state = self.lstm_layer(prev_output, prev_state, input_lstm)
            collect_outputs.append(curr_out)
            prev_output = self.next_location(curr_out, i)
            prev_state = next_state
            
        class_outs = self.fc_layer(collect_outputs[-1], GLIMPSE_FC2, NUM_CLASSES, 'softmax', tf.nn.relu)

        return collect_outputs, tf.nn.softmax(class_outs), self.collect_means, self.collect_locs
    

    # This function is called by the current LSTM cell to get inputs to the next cell
    # next_inputs are of dimension [batch_size, 256]
    def next_location(self, prev_inputs, i):
        with tf.variable_scope('next_loc', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights", [GLIMPSE_FC2, LOC_DIM], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases", [LOC_DIM], initializer=tf.contrib.layers.xavier_initializer())
            y = tf.add(tf.matmul(prev_inputs, weights), biases)
            y = tf.stop_gradient(y)
            
        means = tf.clip_by_value(y, -1., 1.)
        means = tf.stop_gradient(means)

        self.collect_means.append(means)

        dist = tf.contrib.distributions.Normal(means, np.ones((self.batch_size, 2), dtype=np.float32))
        locs = tf.squeeze(tf.clip_by_value(dist.sample([1]), -1.0, 1.0))

        self.collect_locs.append(locs)

        next_inputs = self.glimpse_network(self.inputs, locs)
        
        return next_inputs


    # This function has the glimpse network where the locations are processed 
    # output is a 2D tensor of dimension [batch_size, 256]
    def glimpse_network(self, input_img, locations):

        loc_out1 = self.fc_layer(locations, LOC_DIM, GLIMPSE_FC1, 'lc1', tf.nn.relu)
        loc_out2 = self.fc_layer(loc_out1, GLIMPSE_FC1, GLIMPSE_FC2, 'lc2', tf.nn.relu)

        glimpses = tf.image.extract_glimpse(input_img, [G_WIN_SIZE,G_WIN_SIZE], 
                                            locations, centered=True, normalized=True)

        glimpses = tf.reshape(glimpses, [-1, G_WIN_SIZE*G_WIN_SIZE*IMG_DEPTH])

        g_out1 = self.fc_layer(glimpses, G_DIM, GLIMPSE_FC1, 'g1', tf.nn.relu)
        g_out2 = self.fc_layer(g_out1, GLIMPSE_FC1, GLIMPSE_FC2, 'g2', tf.nn.relu)


        return tf.nn.relu(loc_out2 + g_out2)


    # general template for a fully connected layer used by the model
    # output dimensions are [batch_size, out_size]
    def fc_layer(self, image, in_size, out_size, name, activation):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases", [out_size], initializer=tf.contrib.layers.xavier_initializer())
            y = tf.add(tf.matmul(image, weights), biases)
            tf.summary.histogram('weights_hist', weights)
            tf.summary.histogram('biases_hist', biases)

            if activation is not None:
                y = activation(y)

            return y
    
    
    def lstm_layer(self, last_state, last_outout, curr_input):
        
        with tf.variable_scope('lstm_cell', reuse=tf.AUTO_REUSE):
            whprev = tf.get_variable("whprev", [GLIMPSE_FC2, GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            wx = tf.get_variable("wxcurr", [GLIMPSE_FC2, GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            wf = tf.get_variable("wf", [GLIMPSE_FC2, GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            bf = tf.get_variable("bf", [GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            
            wi = tf.get_variable("wi", [GLIMPSE_FC2, GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            bi = tf.get_variable("bi", [GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            
            wc = tf.get_variable("wc", [GLIMPSE_FC2, GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            bc = tf.get_variable("bc", [GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            
            wo = tf.get_variable("wo", [GLIMPSE_FC2, GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            bo = tf.get_variable("bo", [GLIMPSE_FC2], initializer=tf.contrib.layers.xavier_initializer())
            
            main_mix = tf.matmul(curr_input, whprev)+ tf.matmul(curr_input, wx)
            
            ft = tf.nn.sigmoid(tf.add(tf.matmul(main_mix, wf), bf))
            
            it = tf.nn.sigmoid(tf.add(tf.matmul(main_mix, wi), bi))
            
            cbart = tf.nn.tanh(tf.add(tf.matmul(main_mix, wc), bc))
             
            ct = tf.multiply(ft, last_state) + tf.multiply(it, cbart)
            
            ot = tf.nn.sigmoid(tf.add(tf.matmul(main_mix, wo), bo))
            
            ht = tf.multiply(ot, tf.nn.tanh(ct))
            
            return ht, ct
            
            
        
        