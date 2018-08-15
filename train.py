#-*-coding:utf8 -*-

import os
import time
import numpy as np
import random

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python import debug as tf_debug

import model
import pdb

os.environ['CUDA_VISIBLE_DEVICES']='1'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output','./models',
                          """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_integer('batch_size',32,
                            """Mini-batch size""")
tf.app.flags.DEFINE_float('learning_rate',1e-4,
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum',0.9,
                          """Optimizer gradient first-order momentum""")
tf.app.flags.DEFINE_float('decay_rate',0.9,
                          """Learning rate decay base""")
tf.app.flags.DEFINE_float('decay_steps',1000,
                          """Learning rate decay exponent scale""")
tf.app.flags.DEFINE_float('decay_staircase',False,
                          """Staircase learning rate decay by integer division""")
tf.app.flags.DEFINE_integer('max_num_steps', 30000,
                            """Number of optimization steps to run""")
#tf.logging.set_verbosity(tf.logging.INFO)
# Non-configurable parameters
optimizer='Adam'
 # 'Configure' training mode for dropout layers
# vocab = open('data/vocab.txt').read().split('\n')
vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
num_chars = len(vocab)
vocab2idx = dict([(vocab[i], i) for i in range(len(vocab))])
idx2vocab = dict([(i, vocab[i]) for i in range(len(vocab))])


chars2len = {4: 80, 5: 97, 6: 115, 7: 132, 8: 150}
# by gen_data.py,we get the map of  words's numbers and sample image width
len2chars = {80: 4, 97: 5, 115: 6, 132: 7, 150: 8}


def _parse_single_example(filename,label):
    image_data = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_data,channels=1)
    # new_size = tf.constant([40,150])
    # image_resized = tf.image.resize_images(image_decoded,new_size)

    return image_decoded,label

def _dense2sparse(labels):
    '''
    translate labels to sparseTensor
    '''
    indices = []
    values = []
    for i in range(FLAGS.batch_size):
        for j in range(len(labels[i])):
            indices.append([i,j])
            values.append(vocab2idx[labels[i].decode()[j]])
    dense_shape=[FLAGS.batch_size,len(labels[i])]
    label = tf.SparseTensorValue(indices=indices,values=values,dense_shape=dense_shape)
    
    return label

def _read_data(txt,train_mode):

    filenames = []
    labels = []
    with open('data/'+txt,'r') as f:
        for line in f.readlines():
            filenames.append('data/sample/'+line.split(' ')[0])
            labels.append(line.split(' ')[1][:-1])
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
    dataset = dataset.map(_parse_single_example)
    if train_mode == True:
        dataset = dataset.batch(32).repeat(100)#train dataset,n*epoches
    else:
        dataset = dataset.batch(32)

    iterator = dataset.make_one_shot_iterator()
    batch_data = iterator.get_next()

    return batch_data

def _get_training(rnn_logits,label,sequence_length,global_steps):
    """Set up training ops"""
    with tf.name_scope("train"):

        rnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)#tf.get_collection：从一个集合中取出全部变量，是一个列
        loss = model.ctc_loss_layer(rnn_logits,label,sequence_length) 
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):#控制依赖
            learning_rate = tf.train.exponential_decay(
                                    FLAGS.learning_rate,
                                    global_steps,
                                    FLAGS.decay_steps,
                                    FLAGS.decay_rate,
                                    staircase=FLAGS.decay_staircase,
                                    name='learning_rate')
            optimizer = tf.train.AdamOptimizer(
                                learning_rate=learning_rate,
                                beta1=FLAGS.momentum)
            train_op = tf.contrib.layers.optimize_loss(
                                loss=loss,
                                global_step=global_steps,
                                learning_rate=learning_rate, 
                                optimizer=optimizer,
                                variables=rnn_vars)

            tf.summary.scalar( 'learning_rate', learning_rate )

    return train_op

def _get_session_config():
    """Setup session config to soften device placement"""

    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config

def main(argv=None):
    global_steps = tf.Variable(0)
    image = tf.placeholder(shape=[None,None,None,1],dtype=tf.float32) 
    label = tf.sparse_placeholder(dtype=tf.int32)
    sequence_length = tf.placeholder(dtype=tf.int32)
    train_mode = tf.placeholder(dtype=tf.bool)
    '''
    train
    '''
    features = model.convnet_layers(image, train_mode)
    logits = model.rnn_layers(features, sequence_length,num_chars )
    train_op = _get_training(logits,label,sequence_length,global_steps)
    predictions,_ = tf.nn.ctc_beam_search_decoder(logits, 
                        sequence_length,
                        beam_width=128,
                        top_paths=1,
                        merge_repeated=True)
    res = tf.cast(predictions[0], tf.int32)
    res = tf.sparse_tensor_to_dense(res)   

    session_config = _get_session_config()
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/')
    saver = tf.train.Saver()

    with tf.Session(config=session_config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        step=sess.run(global_steps)

        batch_data_train = {}
        batch_data_val = {}
        for i in range(4,9):
            batch_data_train[i] = _read_data('txt/train_'+str(i)+'.txt',True)
            batch_data_val[i] = _read_data('txt/val_'+str(i)+'.txt',False)
        
        while step<FLAGS.max_num_steps:
            for l in range(4, 9):
                try:
                    batch_data_train1 = sess.run(batch_data_train[l])
                    train_image1 = batch_data_train1[0]
                    train_label1 = batch_data_train1[1]
                    if train_label1.shape[0]!=32:
                        continue
                    train_label2 = _dense2sparse(train_label1)
                    sequence_length1 =  np.ones((FLAGS.batch_size))*len(train_label1[0])
                    step_loss,summary=sess.run([train_op,summary_op],feed_dict={
                                                                    image:train_image1,
                                                                    label:train_label2,
                                                                    sequence_length:sequence_length1,
                                                                    train_mode:True})
                    step=sess.run(global_steps)

                    print("step %d loss is: %f"%(step,step_loss))
                    writer.add_summary(summary,step)
                except tf.errors.OutOfRangeError:
                    saver.save(sess,'models/im2str')
                    print("data is finished!")
                    break
                if step%1000==0:
                    batch_data_val1 = sess.run(batch_data_val[l])
                    val_image1 = batch_data_val1[0]
                    val_label1 = batch_data_val1[1]
                    if val_label1.shape[0]!=32:
                        continue
                    sequence_length1 =  np.ones((FLAGS.batch_size))*len(train_label1[0])
                    res1 = sess.run(res,feed_dict={
                                            image:val_image1,
                                            sequence_length:sequence_length1,
                                            train_mode:True})

                    sq_pred_labels = []
                    sq_true_labels = []
                    for i in range(res1.shape[0]):
                        sq_label = []
                        for j in range(res1.shape[1]):
                            sq_char = idx2vocab[res1[i][j]]
                            sq_label.append(sq_char)
                        sq_true_labels.append(val_label1[i].decode())
                        sq_pred_labels.append(''.join(sq_label))
                    print("#################################")
                    print('True labels',sq_true_labels)
                    print('Pred labels',sq_pred_labels)
                    print("#################################")

                    saver.save(sess,'models/im2str_'+str(step))
                    print('%d step model saved!'%step)

        print("end!")
            

         
                


if __name__ == '__main__':
    tf.app.run()

