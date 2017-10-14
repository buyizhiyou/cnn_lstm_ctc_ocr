#-*-coding:utf8 -*-

import os
import time
import tensorflow as tf
from tensorflow.contrib import learn
import pdb

import data_process
import model

out_charset = "单位NUL/hpμlngmoSceuPd%10E9f2秒 FiDCOIHskM小时或℃A4拷贝个G-万K5WB3月日.867男：女<尿天随机~:≥绝经前—后;+正常饮食低钠卵泡期黄体未怀孕排X岁*^少见检测项目抗梅毒螺旋特异性（T)丙型肝炎(tV乙表面原人免疫缺陷病颜色状蛔虫钩鞭吞噬细胞油滴夏雷登氏脓霉菌孢子红白不消化物隐血试验R有核清晰度外观潘肌钙蛋亚硝酸盐质酮浊胆素葡萄糖管类酵母粘液丝结晶比重上皮电导率值脂酶甲胎铁r癌胚链定心极密固醇）甘三酯无磷高球腺苷脱氨氯a门冬基转移总α岩藻直接b间钾汁氮碱γ谷酰脯二肽酐脑脊反应激同工活羟丁氢乳嗜粒板压积淋巴平均分布宽计数中浓含量国际标准凝对照聚纤维部超敏胱抑y肾滤过全实剩余吸入氧流碳根温纠阴离隙饱和渗透镁碘游促说明沉降淀粉引Y环瓜风湿因溶双J着点列β网织百端利急诊李凡他草鳞相关神元烯角片段枝杆胸水补视合微Ⅰ周本颗快速力旁内载半镜纳紧张浆醛Ⅱ生成胺餐巨腹胰岛绒毛膜线组增殖纺锤动波形尔髓备注析底戊庚非轻λ药克莫司胶殊序骨延长辅助制共读,群空羧露真稀释其糜集，可肠歧酪梭普拉柔嫩感染论呼气钟果帕卧隆κ泌雌刺滑睾幼稚叶层地辛鲎放例酚连ⅣⅢ鉴铜蓝肺支一灰石q幽现症服附纯疱疹弓综出热多去肪的术四粪便规寄、查[房]肿瘤志＋系解谱五功能法输套自费脉泳传研静科两辩呕吐及代谢会荧光室植用残留灶式涂学道种优势身簇所带培养－栓弹图份杂交行儿茶弱阳软淡棕于限褐>黑糊烂绿块浑建议复详亮暗澈±硬干扰严文报告腔仅供参考米|"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model','../data/model',
                          """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('output','test',
                          """Sub-directory of model for test summary events""")
tf.app.flags.DEFINE_integer('batch_size',20,
                            """Eval batch size""")
tf.app.flags.DEFINE_integer('test_interval_secs', 10,
                             'Time between test runs')
tf.app.flags.DEFINE_string('device','/gpu:0',
                           """Device for graph placement""")
tf.app.flags.DEFINE_string('test_path','../data/',
                           """Base directory for test/validation data""")
tf.app.flags.DEFINE_string('filename_pattern','val/words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',4,
                          """Number of readers for input data""")
tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers


def _get_input():
    """Set up and return image, label, width and text tensors"""

    image,width,label,length,text,filename=data_process.threaded_input_pipeline(
        FLAGS.test_path,
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_input_threads,
        num_epochs=None, # Repeat for streaming
        batch_device=FLAGS.device, 
        preprocess_device=FLAGS.device )
    
    return image,width,label,length

def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config

def _get_testing(rnn_logits,sequence_length,label,label_length):
    """Create ops for testing (all scalars): 
       loss: CTC loss function value, 
       label_error:  Batch-normalized edit distance on beam search max
       sequence_error: Batch-normalized sequence error rate
    """
    with tf.name_scope('train'):
        loss = model.ctc_loss_layer(rnn_logits,label,sequence_length)
    with tf.name_scope("test"):
        predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits, 
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=True)
        hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance
        label_errors = tf.edit_distance(hypothesis, label, normalize=False)#计算序列之间的Levenshtein 距离
        sequence_errors = tf.count_nonzero(label_errors,axis=0)#计算非0个数
        total_label_error = tf.reduce_sum( label_errors )
        total_labels = tf.reduce_sum( label_length )
        label_error = tf.truediv( total_label_error, 
                                  tf.cast(total_labels, tf.float32 ),
                                  name='label_error')#对应位置元素的除法运算
        sequence_error = tf.truediv( tf.cast( sequence_errors, tf.int32 ),
                                     tf.shape(label_length)[0],
                                     name='sequence_error')
        tf.summary.scalar( 'loss', loss )
        tf.summary.scalar( 'label_error', label_error )
        tf.summary.scalar( 'sequence_error', sequence_error )

    return loss, label_error, sequence_error,predictions,label_errors

def _get_checkpoint():
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path=ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path

def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) + 
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn


def main(argv=None):

    with tf.Graph().as_default():
        image,width,label,length = _get_input()#读取数据
        #image,width,label,length = ?
        with tf.device(FLAGS.device):
            features,sequence_length = model.convnet_layers( image, width, mode)
            logits = model.rnn_layers( features, sequence_length,
                                       data_process.num_classes() )
            loss,label_error,sequence_error,predictions,label_errors = _get_testing(
                logits,sequence_length,label,length)

        global_step = tf.contrib.framework.get_or_create_global_step()

        session_config = _get_session_config()
        restore_model = _get_init_trained()#restore model
        
        summary_op = tf.summary.merge_all()
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        summary_writer = tf.summary.FileWriter( os.path.join(FLAGS.model,
                                                            FLAGS.output) )

        step_ops = [global_step, loss, label_error, sequence_error]

        with tf.Session(config=session_config) as sess:   
            sess.run(init_op)
            coord = tf.train.Coordinator() # Launch reader threads
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            summary_writer.add_graph(sess.graph)
            try:            
                while True:
                    restore_model(sess, _get_checkpoint()) # Get latest checkpoint

                    if not coord.should_stop():
                        #pdb.set_trace()
                        step_vals = sess.run(step_ops)
                        # image = sess.run(image)#shape:(20, 65, 256, 1) 0.5
                        # width = sess.run(width)#array([[256],[256],...[256]])
                        # label_errors = sess.run(label_errors)#array([ 1.,  1.,  5.,  1.,  1.,  0.,  2.,  0.,  2.,  3.,  1.,  1.,  1.,1.,  5.,  1.,  2.,  3.,  0.,  2.], dtype=float32)
                        # sequence_length = sess.run(sequence_length)#array([29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,29, 29, 29], dtype=int32)
                        # length = sess.run(length)#shape:(20, ),array([[ 2],[ 4],[ 8],[12]...,[ 7],[ 4],[ 4]])
                        # logits = sess.run(logits)#shape:(29, 20, 627)
                        # label = sess.run(label)#SparseTensor,(indices,values,shape=[20,*])
                        # predictions = sess.run(predictions[0])#SparseTensor,[indices,values,shape=[20,*]]
                        # label2 = tf.sparse_tensor_to_dense(label)
                        # predictions2 = tf.sparse_tensor_to_dense(predictions)
                        # label2 = sess.run(label2)
                        # predictions2 = sess.run(predictions2)
                        
                        print ("loss:%f,label_error:%f,sequence_error:%f"%(step_vals[1],step_vals[2],step_vals[3]))
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str,step_vals[0])
                    else:
                        break
                    time.sleep(FLAGS.test_interval_secs)
            except tf.errors.OutOfRangeError:
                    print('Done')
            finally:
                coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
