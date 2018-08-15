#-*-coding:utf8 -*-

import tensorflow as tf
from tensorflow.contrib import learn
import pdb


layer_params = [ [ 64,  3, 'same',  'conv1', False], 
                 [ 128, 3, 'same',  'conv2', False],
                 [ 256, 3, 'same',  'conv3', False],
                 [ 256, 3, 'same',  'conv4', False],
                 [ 512, 3, 'same',  'conv5', True], 
                 [ 512, 3, 'same',  'conv6', True],
                 [ 512, 3, 'valid',  'conv7', False],]
rnn_size = 128
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

def norm_layer( bottom, training, name):
    """Short function to build a batch normalization layer with less syntax"""
    top = tf.layers.batch_normalization( bottom, axis=3, # channels last,
                                         training=training,
                                         name=name )
    return top


def convnet_layers(inputs, training):
    """Build convolutional network layers attached to the given input tensor"""


    # inputs should have shape [ ?, 32, ?, 1 ]
    with tf.variable_scope("convnet"): # h,w
        # inputs = tf.random_uniform((1,40,150,1))
        conv1 = conv_layer(inputs, layer_params[0], training )
        pool1 = tf.layers.max_pooling2d(conv1,[2,2],[2,2] ,padding='same', name='pool1')

        conv2 = conv_layer(pool1,layer_params[1], training )
        pool2 = tf.layers.max_pooling2d(conv2,[2,2],[2,2], padding='same', name='pool2')

        conv3 = conv_layer(pool2,layer_params[2], training )
        conv4 = conv_layer(conv3,layer_params[3], training )
        pool3 = tf.layers.max_pooling2d(conv4,[2,2],[2,2] ,padding='same', name='pool3')

        conv5 = conv_layer(pool3,layer_params[4], training )
        conv6 = conv_layer(conv5,layer_params[5], training )
        pool4 = tf.layers.max_pooling2d(conv6, [2,1], [2,1], padding='same', name='pool4')

        conv7 = conv_layer(pool4,layer_params[6], training )
        features = tf.squeeze(conv7, axis=1, name='features') #数据降维，只裁剪等于1的维度


        # two = tf.constant(2, dtype=tf.int32, name='two')
        # after_pool1 = tf.floor_div(widths,two) 
        # after_pool2 = tf.floor_div(after_pool1,two)
        # after_pool3 = tf.floor_div(after_pool2,two)
        # sequence_length = tf.reshape(after_pool3,[-1], name='seq_len') # Vectorize

        return features

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
                           ignore_longer_outputs_than_inputs=True,time_major=True )
    total_loss = tf.reduce_mean(loss)
    return total_loss
