#-*-coding:utf8 -*-

import tensorflow as tf
from tensorflow.contrib import learn

# Layer params:   Filts K  Padding  Name     BatchNorm?
# layer_params = [ 
#                  [  64, 3, 'same',  'conv1', False], 
#                  [  64, 3, 'same',  'conv2', True], # pool
#                  [ 128, 3, 'same',  'conv3', False], 
#                  [ 128, 3, 'same',  'conv4', True], # hpool
#                  [ 256, 3, 'same',  'conv5', False],
#                  [ 256, 3, 'same',  'conv6', True], # hpool
#                  [ 512, 3, 'same',  'conv7', False], 
#                  [ 512, 3, 'same',  'conv8', True]] # hpool 3

layer_params = [ 
                 [  128, 3, 'same',  'conv1', True], 
                 [  64, 3, 'same',  'conv2', True], 

                 [  64, 3, 'valid',  'conv3', False], 
                 [  64, 3, 'same',  'conv4', True], # pool
                 [ 128, 3, 'same',  'conv5', False], 
                 [ 128, 3, 'same',  'conv6', True], # hpool
                 [ 256, 3, 'same',  'conv7', False],
                 [ 256, 3, 'same',  'conv8', True], # hpool
                 [ 512, 3, 'same',  'conv9', False], 
                 [ 512, 3, 'same',  'conv10', True]] # hpool 3

rnn_size = 2**9
dropout_rate = 0.5

def conv_layer(bottom, params, training ):
    """Build a convolutional layer using entry from layer_params)"""

    batch_norm = params[4] # Boolean

    if batch_norm:
        activation=None
    else:
        activation=tf.nn.relu

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    top = tf.layers.conv2d(bottom, 
                           filters=params[0],
                           kernel_size=params[1],
                           padding=params[2],
                           activation=activation,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           name=params[3])
    if batch_norm:
        top = norm_layer( top, training, params[3]+'/batch_norm' )
        top = tf.nn.relu( top, name=params[3]+'/relu' )

    return top

def pool_layer( bottom, v, h, padding, name ):
    """Short function to build a pooling layer with less syntax"""
    top = tf.layers.max_pooling2d( bottom, 2, [v,h], 
                                   padding=padding, 
                                   name=name)
    return top

def norm_layer( bottom, training, name):
    """Short function to build a batch normalization layer with less syntax"""
    top = tf.layers.batch_normalization( bottom, axis=3, # channels last,
                                         training=training,
                                         name=name )
    return top


def convnet_layers(inputs, widths, mode):
    """Build convolutional network layers attached to the given input tensor"""

    training = (mode == learn.ModeKeys.TRAIN)

    # inputs should have shape [ ?, 32, ?, 1 ]
    with tf.variable_scope("convnet"): # h,w
        
        # conv1 = conv_layer(inputs, layer_params[0], training ) # 30,30
        # conv2 = conv_layer( conv1, layer_params[1], training ) # 30,30
        # pool2 = pool_layer( conv2, 2, 'valid', 'pool2')        # 15,15
        # conv3 = conv_layer( pool2, layer_params[2], training ) # 15,15
        # conv4 = conv_layer( conv3, layer_params[3], training ) # 15,15
        # pool4 = pool_layer( conv4, 1, 'valid', 'pool4' )       # 7,14
        # conv5 = conv_layer( pool4, layer_params[4], training ) # 7,14
        # conv6 = conv_layer( conv5, layer_params[5], training ) # 7,14
        # pool6 = pool_layer( conv6, 1, 'valid', 'pool6')        # 3,13
        # conv7 = conv_layer( pool6, layer_params[6], training ) # 3,13
        # conv8 = conv_layer( conv7, layer_params[7], training ) # 3,13
        # pool8 = tf.layers.max_pooling2d( conv8, [3,1], [3,1], 
        #                            padding='valid', name='pool8') # 1,13
        conv1 = conv_layer(inputs, layer_params[0], training ) # 64,256
        pool1 = pool_layer( conv1, 2,2 ,'valid', 'pool1')        # 32,128
        conv2 = conv_layer( pool1, layer_params[1], training ) # 32,64
        pool2 = pool_layer( conv2, 1,2 , 'valid', 'pool2')        # 32,64

        conv3 = conv_layer(pool2, layer_params[2], training ) # 30,62
        conv4 = conv_layer( conv3, layer_params[3], training ) # 30,62
        pool3 = pool_layer( conv4, 2,2, 'valid', 'pool3')        # 15,31
        conv5 = conv_layer( pool3, layer_params[4], training ) # 15,31
        conv6 = conv_layer( conv5, layer_params[5], training ) # 15,31
        pool4 = pool_layer( conv6, 2,1 ,'valid', 'pool4' )       # 7,30
        conv7 = conv_layer( pool4, layer_params[6], training ) # 7,30
        conv8 = conv_layer( conv7, layer_params[7], training ) # 7,30
        pool5 = pool_layer( conv8, 2,1 ,'valid', 'pool5')        # 3,29
        conv9 = conv_layer( pool5, layer_params[8], training ) # 3,29
        conv10 = conv_layer( conv9, layer_params[9], training ) # 3,29
        pool6 = tf.layers.max_pooling2d( conv10, [3,1], [3,1], 
                                   padding='valid', name='pool6') # 1,29
        features = tf.squeeze(pool6, axis=1, name='features') #数据降维，只裁剪等于1的维度

        kernel_sizes = [ params[1] for params in layer_params]

        # Calculate resulting sequence length from original image widths
        # conv1_trim = tf.constant( 2 * (kernel_sizes[0] // 2),
        #                           dtype=tf.int32,
        #                           name='conv1_trim')  #2
        one = tf.constant(1, dtype=tf.int32, name='one')
        two = tf.constant(2, dtype=tf.int32, name='two')
        # widths = tf.floor_div(widths, 8 )#x//8  256//8
        # after_conv1 = tf.subtract( widths, conv1_trim)#x-2
        # after_pool2 = tf.floor_div( after_conv1, two )#x//2
        # after_pool4 = tf.subtract(after_pool2, two)#x-2
        # sequence_length = tf.reshape(after_pool4,[-1], name='seq_len') # Vectorize
        after_pool1 = tf.floor_div(widths,two) 
        after_pool2 = tf.floor_div(after_pool1,two)
        after_conv3 = tf.subtract(after_pool2,two)
        after_pool3 = tf.floor_div(after_conv3,two)
        after_pool4 = tf.subtract(after_pool3,one)
        after_pool5 = tf.subtract(after_pool4,one)
        sequence_length = tf.reshape(after_pool5,[-1], name='seq_len') # Vectorize

        return features,sequence_length

def rnn_layer(bottom_sequence,sequence_length,rnn_size,scope):
    """Build bidirectional (concatenated output) RNN layer"""

    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    
    # Default activation is tanh
    cell_fw = tf.contrib.rnn.LSTMCell( rnn_size, 
                                       initializer=weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell( rnn_size, 
                                       initializer=weight_initializer)
    #drop_out
    cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw, 
                                            input_keep_prob=dropout_rate )
    cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw, 
                                            input_keep_prob=dropout_rate )
    
    rnn_output,_ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, bottom_sequence,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.float32,
        scope=scope)
    
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    rnn_output_stack = tf.concat(rnn_output,2,name='output_stack')
    
    return rnn_output_stack


def rnn_layers(features, sequence_length, num_classes):
    """Build a stack of RNN layers from input features"""

    # Input features is [batchSize paddedSeqLen numFeatures]
    logit_activation = tf.nn.relu
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope("rnn"):
        # Transpose to time-major order for efficiency
        rnn_sequence = tf.transpose(features, perm=[1, 0, 2], name='time_major')
        rnn1 = rnn_layer(rnn_sequence, sequence_length, rnn_size, 'bdrnn1')
        rnn2 = rnn_layer(rnn1, sequence_length, rnn_size, 'bdrnn2')
        rnn_logits = tf.layers.dense( rnn2, num_classes+1, 
                                      activation=logit_activation,
                                      kernel_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      name='logits')
        return rnn_logits
    

def ctc_loss_layer(rnn_logits, sequence_labels, sequence_length):
    """Build CTC Loss layer for training"""

    loss = tf.nn.ctc_loss( sequence_labels, rnn_logits, sequence_length,
                           time_major=True )
    total_loss = tf.reduce_mean(loss)
    return total_loss
