#-*-coding:utf8 -*-

import os
import time
import tensorflow as tf
from tensorflow.contrib import learn
import pdb
import cv2
import numpy as np

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

def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) + 
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn


def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config

def predict(path):

    with tf.Graph().as_default():

        image = cv2.imread(path,-1)
        image = np.reshape(image,[64,256,1])
        width = image.shape[1]
        # Rescale from uint8([0,255]) to float([-0.5,0.5])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.subtract(image, 0.5) #减法

        # Pad with copy of first row to expand to 32 pixels height
        first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
        image = tf.concat([first_row, image], 0)
        image = tf.reshape(image,[1,65,256,1])

        with tf.device(FLAGS.device):
            features,sequence_length = model.convnet_layers( image, width, mode)
            rnn_logits = model.rnn_layers( features, sequence_length,
                                       data_process.num_classes() )
            predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits, 
                                       sequence_length,
                                       beam_width=128,
                                       top_paths=1,
                                       merge_repeated=True)
        hypothesis = tf.cast(predictions[0], tf.int32) 
        global_step = tf.contrib.framework.get_or_create_global_step()
        session_config = _get_session_config()
        restore_model = _get_init_trained()#restore model
        
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        with tf.Session(config = session_config) as sess:   
            sess.run(init_op)
            restore_model(sess, ckpt_path='../data/model/model.ckpt-30000') # Get latest checkpoint
            label = sess.run(hypothesis)
            label = tf.sparse_tensor_to_dense(label)
            label = sess.run(label)
            label = list(label[0])
            result = []
            for i in label:
                result.append(out_charset[i])
            result = ''.join(result)

    return result

  
  
if __name__ == '__main__':

    g = open('result.txt','w')
    with open('test.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            filename = line.split(' ')[0]
            path = '../data/raw/'+filename
            result = predict(path)
            g.write(result+'\n')
    g.close()

