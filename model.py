class Model:
    def __init__(self, inputs, b_size):
        self.inputs= inputs
        self.batch_size = b_size
        

    def __call__(self):
        initial_means = tf.random_uniform([batch_size, LOC_DIM], minval=-1, maxval=1)
        
        input_lstm = self.glimpse_network(self.inputs, initial_locs)

        lstm_cell  = tf.nn.rnn_cell.LSTMCell(LSTM_HIDDEN, state_is_tuple=True)
        init_state = lstm_cell.zero_state(batch_size, tf.float32)

        extend_inputs = np.zeros((NUM_GLIMPSES, batch_size, GLIMPSE_FC2))
        extend_inputs[0] = input_lstm

        outputs, state = tf.contrib.legacy_seq2seq.rnn_decoder(extend_inputs, init_state, 
                                                               lstm_cell, loop_function=next_location)
        
        class_outs = self.fc_layer(g_out1, GLIMPSE_FC1, GLIMPSE_FC2, 'g2', tf.nn.relu)

        return class_outs
    

    def next_location(self, prev_inputs, i):
        
        with tf.variable_scope('next_loc'):
            weights = tf.get_variable("weights", [GLIMPSE_FC2, LOC_DIM], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases", [LOC_DIM], initializer=tf.contrib.layers.xavier_initializer())
            y = tf.add(tf.matmul(image, weights), biases)
            y = tf.stop_gradient(y)
            
        locs = tf.clip_by_value(next_loc, -1., 1.)
        locs = tf.stop_gradient(next_loc)

        next_inputs = self.glimpse_network(self.inputs, locs)

        return next_inputs


    def glimpse_network(self, input_img, locations):

        loc_out1 = self.fc_layer(locations, LOC_DIM, GLIMPSE_FC1, 'lc1', tf.nn.relu)
        loc_out2 = self.fc_layer(loc_out1, GLIMPSE_FC1, GLIMPSE_FC2, 'lc2', tf.nn.relu)

        glimpses = tf.image.extract_glimpse(input_img, [G_WIN_SIZE,G_WIN_SIZE], 
                                            locations, centered=True, normalized=True)

        glimpses = tf.reshape(glimpses, [-1, G_WIN_SIZE*G_WIN_SIZE*IMG_DEPTH])

        g_out1 = self.fc_layer(glimpses, G_DIM, GLIMPSE_FC1, 'g1', tf.nn.relu)
        g_out2 = self.fc_layer(g_out1, GLIMPSE_FC1, GLIMPSE_FC2, 'g2', tf.nn.relu)

        return tf.nn.relu(loc_out2 + g_out2)


    def fc_layer(self, image, in_size, out_size, name, activation):
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases", [out_size], initializer=tf.contrib.layers.xavier_initializer())
            y = tf.add(tf.matmul(image, weights), biases)
            tf.summary.histogram('weights_hist', weights)
            tf.summary.histogram('biases_hist', biases)

            if activation is not None:
                y = activation(y)

            return y
    
        