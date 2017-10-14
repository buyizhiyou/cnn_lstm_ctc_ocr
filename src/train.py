#-*-coding:utf8 -*-

import os
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python import debug as tf_debug

import data_process
import model
import pdb


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output','../data/model',
                          """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_string('tune_from','',
                          """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_scope','',
                          """Variable scope for training""")
tf.app.flags.DEFINE_integer('batch_size',20,
                            """Mini-batch size""")
tf.app.flags.DEFINE_float('learning_rate',1e-4,
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum',0.9,
                          """Optimizer gradient first-order momentum""")
tf.app.flags.DEFINE_float('decay_rate',0.9,
                          """Learning rate decay base""")
tf.app.flags.DEFINE_float('decay_steps',5000,
                          """Learning rate decay exponent scale""")
tf.app.flags.DEFINE_float('decay_staircase',False,
                          """Staircase learning rate decay by integer division""")
tf.app.flags.DEFINE_integer('max_num_steps', 30000,
                            """Number of optimization steps to run""")
tf.app.flags.DEFINE_string('train_device','/gpu:1',
                           """Device for training graph placement""")
tf.app.flags.DEFINE_string('input_device','/gpu:0',
                           """Device for preprocess/batching graph placement""")
tf.app.flags.DEFINE_string('train_path','../data/train/',
                           """Base directory for training data""")
tf.app.flags.DEFINE_string('filename_pattern','words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',4,
                          """Number of readers for input data""")
tf.app.flags.DEFINE_integer('width_threshold',None,
                            """Limit of input image width""")
tf.app.flags.DEFINE_integer('length_threshold',None,
                            """Limit of input string length width""")
#tf.logging.set_verbosity(tf.logging.INFO)

# Non-configurable parameters
optimizer='Adam'
mode = learn.ModeKeys.TRAIN # 'Configure' training mode for dropout layers

def _get_input():
    """Set up and return image, label, and image width tensors"""
    #pdb.set_trace()
    image,width,label,_,_,_=data_process.bucketed_input_pipeline(
        FLAGS.train_path, 
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size, 
        num_threads=FLAGS.num_input_threads,
        input_device=FLAGS.input_device,
        width_threshold=FLAGS.width_threshold,
        length_threshold=FLAGS.length_threshold )

    #tf.summary.image('images',image) 
    return image,width,label

# def _get_single_input():
#     """Set up and return image, label, and width tensors"""

#     image,width,label,length,text,filename=data_process.threaded_input_pipeline(
#         deps.get('records'), 
#         str.split(FLAGS.filename_pattern,','),
#         batch_size=1,
#         num_threads=FLAGS.num_input_threads,
#         num_epochs=1,
#         batch_device=FLAGS.input_device, 
#         preprocess_device=FLAGS.input_device )
#     return image,width,label,length,text,filename

def _get_training(rnn_logits,label,sequence_length):
    """Set up training ops"""
    with tf.name_scope("train"):

        if FLAGS.tune_scope:
            scope=FLAGS.tune_scope
        else:
            scope="convnet|rnn"

        rnn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope)#tf.get_collection：从一个集合中取出全部变量，是一个列
        loss = model.ctc_loss_layer(rnn_logits,label,sequence_length) 
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):

            learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate,
                tf.train.get_global_step(),
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                staircase=FLAGS.decay_staircase,
                name='learning_rate')

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=FLAGS.momentum)
            
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
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

def _get_init_pretrained():
    """Return lambda for reading pretrained initial model"""

    if not FLAGS.tune_from:
        return None
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    ckpt_path=FLAGS.tune_from
    init_fn = lambda sess: saver_reader.restore(sess, ckpt_path)

    return init_fn


def main(argv=None):

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        image,width,label = _get_input()

        with tf.device(FLAGS.train_device):#三个阶段的网络
            features,sequence_length = model.convnet_layers( image, width, mode)
            logits = model.rnn_layers( features, sequence_length,
                                       data_process.num_classes() )
            train_op = _get_training(logits,label,sequence_length)

        session_config = _get_session_config()

        summary_op = tf.summary.merge_all()
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 
        sv = tf.train.Supervisor(
            logdir=FLAGS.output,#会自动去logdir中去找checkpoint，如果没有的话，自动执行初始化
            init_op=init_op,
            summary_op=summary_op,
            save_summaries_secs=30,
            save_model_secs=150)
# Supervisor帮助我们处理一些事情
# （1）自动去checkpoint加载数据或初始化数据
# （2）自身有一个Saver，可以用来保存checkpoint
# （3）有一个summary_computed用来保存Summary
# 所以，我们就不需要：
# （1）手动初始化或从checkpoint中加载数据
# （2）不需要创建Saver，使用sv内部的就可以
# （3）不需要创建summary writer


        with sv.managed_session(config=session_config) as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            step = sess.run(global_step)
            # pdb.set_trace()
            # image = sess.run(image)#image.shape(20, 65, 256, 1),0.5

            while step < FLAGS.max_num_steps:
                if sv.should_stop():
                    break        
                # features = sess.run(features)#shape:(20, 29, 512)
                # sequence_length = sess.run(sequence_length)#shape:(20,) ,29
                # logits = sess.run(logits)#shape:(29, 20, 627) 
                # label = sess.run(label)#SparseTensor==>(indices,values,shape) dense_shape=array([20,13])        
                
                [step_loss,step]=sess.run([train_op,global_step])
                print("step %d loss is: %f"%(step,step_loss))
            sv.saver.save( sess, os.path.join(FLAGS.output,'model.ckpt'),
                           global_step=global_step)


if __name__ == '__main__':
    tf.app.run()
