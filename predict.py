#-*-coding:utf8 -*-

import os
import time
import tensorflow as tf
from tensorflow.contrib import learn
import pdb
import cv2
import numpy as np
import skimage.io as io 

import model


os.environ['CUDA_VISIBLE_DEVICES']='1'

vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
num_chars = len(vocab)
vocab2idx = dict([(vocab[i], i) for i in range(len(vocab))])
idx2vocab = dict([(i, vocab[i]) for i in range(len(vocab))])

chars2len = {4: 80, 5: 97, 6: 115, 7: 132, 8: 150}
# by gen_data.py,we get the map of  words's numbers and sample image width
len2chars = {80: 4, 97: 5, 115: 6, 132: 7, 150: 8}

def restore_model(sess):
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver()
    model_file=tf.train.latest_checkpoint('models/')
    print("load model file:",model_file)
    saver_reader.restore(sess, model_file)

def predict(path):
    im = io.imread(path,as_grey=True)*255
    width = im.shape[1]
    sequence_length = [len2chars[width]]*32
    im = np.expand_dims(im,0)
    im = np.expand_dims(im,3)
    im = np.repeat(im,32,axis=0)
    im = tf.constant(im,dtype=tf.float32)
    sequence_length = tf.constant(sequence_length)
    features = model.convnet_layers(im, False)
    rnn_logits = model.rnn_layers(features, sequence_length,num_chars)
    predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits, 
                                sequence_length,
                                beam_width=128,
                                top_paths=1,
                                merge_repeated=True)
    hypothesis = tf.cast(predictions[0], tf.int32) 
    label = tf.sparse_tensor_to_dense(hypothesis)
    
    init_op = tf.group( tf.global_variables_initializer(),
                        tf.local_variables_initializer()) 

    with tf.Session() as sess:   
        sess.run(init_op)
        restore_model(sess) # Get latest checkpoint
        label = sess.run(label)
        label = list(label[0])
        result = []
        for i in label:
            result.append(idx2vocab[i])
        result = ''.join(result)

    return result

if __name__ == '__main__':

    print(predict('788_1.jpg'))
