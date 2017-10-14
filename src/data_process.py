#-*- coding:utf8 -*-

import os
import tensorflow as tf
import pdb

# The list (well, string) of valid output characters
# If any example contains a character not found here, an error will result
# from the calls to .index in the decoder below
#out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
out_charset = "单位NUL/hpμlngmoSceuPd%10E9f2秒 FiDCOIHskM小时或℃A4拷贝个G-万K5WB3月日.867男：女<尿天随机~:≥绝经前—后;+正常饮食低钠卵泡期黄体未怀孕排X岁*^少见检测项目抗梅毒螺旋特异性（T)丙型肝炎(tV乙表面原人免疫缺陷病颜色状蛔虫钩鞭吞噬细胞油滴夏雷登氏脓霉菌孢子红白不消化物隐血试验R有核清晰度外观潘肌钙蛋亚硝酸盐质酮浊胆素葡萄糖管类酵母粘液丝结晶比重上皮电导率值脂酶甲胎铁r癌胚链定心极密固醇）甘三酯无磷高球腺苷脱氨氯a门冬基转移总α岩藻直接b间钾汁氮碱γ谷酰脯二肽酐脑脊反应激同工活羟丁氢乳嗜粒板压积淋巴平均分布宽计数中浓含量国际标准凝对照聚纤维部超敏胱抑y肾滤过全实剩余吸入氧流碳根温纠阴离隙饱和渗透镁碘游促说明沉降淀粉引Y环瓜风湿因溶双J着点列β网织百端利急诊李凡他草鳞相关神元烯角片段枝杆胸水补视合微Ⅰ周本颗快速力旁内载半镜纳紧张浆醛Ⅱ生成胺餐巨腹胰岛绒毛膜线组增殖纺锤动波形尔髓备注析底戊庚非轻λ药克莫司胶殊序骨延长辅助制共读,群空羧露真稀释其糜集，可肠歧酪梭普拉柔嫩感染论呼气钟果帕卧隆κ泌雌刺滑睾幼稚叶层地辛鲎放例酚连ⅣⅢ鉴铜蓝肺支一灰石q幽现症服附纯疱疹弓综出热多去肪的术四粪便规寄、查[房]肿瘤志＋系解谱五功能法输套自费脉泳传研静科两辩呕吐及代谢会荧光室植用残留灶式涂学道种优势身簇所带培养－栓弹图份杂交行儿茶弱阳软淡棕于限褐>黑糊烂绿块浑建议复详亮暗澈±硬干扰严文报告腔仅供参考米|"

def num_classes():
    return len(out_charset)#626

def bucketed_input_pipeline(base_dir,
                            file_patterns,
                            num_threads=4,
                            batch_size=32,
                            boundaries=[32, 64, 96, 128, 160, 192, 224, 256],
                            input_device=None,
                            width_threshold=None,
                            length_threshold=None,
                            num_epochs=None):
    """Get input tensors bucketed by image width
    Returns:
      image : float32 image tensor [batch_size 32 ? 1] padded to batch max width
      width : int32 image widths (for calculating post-CNN sequence length)
      label : Sparse tensor with label sequences for the batch
      length : Length of label sequence (text length)
      text  :  Human readable string for the image
      filename : Source file path
    """
    queue_capacity = num_threads*batch_size*2  #4*32*2 = 256
    # Allow a smaller final batch if we are going for a fixed number of epochs
    final_batch = (num_epochs!=None) 

    data_queue = _get_data_queue(base_dir, file_patterns, 
                                 capacity=queue_capacity,
                                 num_epochs=num_epochs) #返回要读取的文件列表

    with tf.device(input_device): # Create bucketing batcher
        image, width, label, length, text, filename  = _read_word_record(data_queue)#一条一条去读
        image = _preprocess_image(image) # 对返回的图像预处理

        keep_input = _get_input_filter(width, width_threshold,
                                       length, length_threshold)#判断是否保留
        data_tuple = [image, label, length, text, filename]
        width,data_tuple = tf.contrib.training.bucket_by_sequence_length(
            input_length=width,
            tensors=data_tuple,
            bucket_boundaries=boundaries,
            batch_size=batch_size,
            capacity=queue_capacity,
            keep_input=keep_input,
            allow_smaller_final_batch=final_batch,
            dynamic_pad=True)
        [image, label, length, text, filename] = data_tuple
        label = tf.deserialize_many_sparse(label, tf.int64) # post-batching...将多个稀疏的serialized_sparse合并成一个
        label = tf.cast(label, tf.int32) # for ctc_loss
    return image, width, label, length, text, filename

def threaded_input_pipeline(base_dir,file_patterns,
                            num_threads=4,
                            batch_size=20,
                            batch_device=None,
                            preprocess_device=None,
                            num_epochs=None):

    queue_capacity = num_threads*batch_size*2
    # Allow a smaller final batch if we are going for a fixed number of epochs
    final_batch = (num_epochs!=None) 

    data_queue = _get_data_queue(base_dir, file_patterns, 
                                 capacity=queue_capacity,
                                 num_epochs=num_epochs)

    # each thread has a subgraph with its own reader (sharing filename queue)
    data_tuples = [] # list of subgraph [image, label, width, text] elements
    with tf.device(preprocess_device):
        for _ in range(num_threads):
            image, width, label, length, text, filename  = _read_word_record(data_queue)
            image = _preprocess_image(image) #
            data_tuples.append([image, width, label, length, text, filename])

    with tf.device(batch_device): # Create batch queue
        image, width, label, length, text, filename  = tf.train.batch_join( 
            data_tuples, 
            batch_size=batch_size,
            capacity=queue_capacity,
            allow_smaller_final_batch=final_batch,
            dynamic_pad=True)
        label = tf.deserialize_many_sparse(label, tf.int64) # post-batching...
        label = tf.cast(label, tf.int32) # for ctc_loss
    return image, width, label, length, text, filename

def _get_input_filter(width, width_threshold, length, length_threshold):
    """Boolean op for discarding input data based on string or image size
    Input:
      width            : Tensor representing the image width
      width_threshold  : Python numerical value (or None) representing the 
                         maximum allowable input image width 
      length           : Tensor representing the ground truth string length
      length_threshold : Python numerical value (or None) representing the 
                         maximum allowable input string length
   Returns:
      keep_input : Boolean Tensor indicating whether to keep a given input 
                  with the specified image width and string length
"""

    keep_input = None

    if width_threshold!=None:
        keep_input = tf.less_equal(width, width_threshold)#<=

    if length_threshold!=None:
        length_filter = tf.less_equal(length, length_threshold)#<=
        if keep_input==None:
            keep_input = length_filter 
        else:
            keep_input = tf.logical_and( keep_input, length_filter)

    if keep_input==None:
        keep_input = True
    else:
        keep_input = tf.reshape( keep_input, [] ) # explicitly make a scalar

    return keep_input#bool

def _get_data_queue(base_dir, file_patterns=['*.tfrecord'], capacity=2**15,
                    num_epochs=None):
    """Get a data queue for a list of record files"""

    # List of lists ...
    data_files = [tf.gfile.Glob(os.path.join(base_dir,file_pattern))
                  for file_pattern in file_patterns]  #[['../data/train/words-000.tfrecord']]
    # flatten
    data_files = [data_file for sublist in data_files for data_file in sublist]#['../data/train/words-000.tfrecord']
    data_queue = tf.train.string_input_producer(data_files, 
                                                capacity=capacity,
                                                num_epochs=num_epochs) #<tensorflow.python.ops.data_flow_ops.FIFOQueue object at 0x7fc4289383c8>
    return data_queue

def _read_word_record(data_queue):

    reader = tf.TFRecordReader() # Construct a general reader
    key, example_serialized = reader.read(data_queue) 
    #<tf.Tensor 'ReaderReadV2:0' shape=() dtype=string>
    #<tf.Tensor 'ReaderReadV2:1' shape=() dtype=string>
    feature_map = {
        'image/encoded':  tf.FixedLenFeature( [], dtype=tf.string, 
                                              default_value='' ),
        'image/labels':   tf.VarLenFeature( dtype=tf.int64 ), 
        'image/width':    tf.FixedLenFeature( [1], dtype=tf.int64,
                                              default_value=1 ),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value='' ),
        'text/string':     tf.FixedLenFeature([], dtype=tf.string,
                                             default_value='' ),
        'text/length':    tf.FixedLenFeature( [1], dtype=tf.int64,
                                              default_value=1 )
    }
    features = tf.parse_single_example( example_serialized, feature_map )##调用parse_single_example函数解析数据
    # {'text/length': <tf.Tensor 'ParseSingleExample/Squeeze_text/length:0' shape=(1,) dtype=int64>, 
    # 'image/width': <tf.Tensor 'ParseSingleExample/Squeeze_image/width:0' shape=(1,) dtype=int64>, 
    # 'image/filename': <tf.Tensor 'ParseSingleExample/Squeeze_image/filename:0' shape=() dtype=string>, 
    # 'text/string': <tf.Tensor 'ParseSingleExample/Squeeze_text/string:0' shape=() dtype=string>, 
    # 'image/encoded': <tf.Tensor 'ParseSingleExample/Squeeze_image/encoded:0' shape=() dtype=string>,
    # 'image/labels': <tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x7fc42894e0b8>}
    image = tf.image.decode_jpeg( features['image/encoded'], channels=1 ) #toMatrix
    width = tf.cast( features['image/width'], tf.int32) # for ctc_loss
    #pdb.set_trace()
    label = tf.serialize_sparse( features['image/labels'] ) #返回一个字符串的3-vector（1-D的tensor），分别表示索引、值、shape,返回稀疏向量
    length = features['text/length']
    text = features['text/string']
    filename = features['image/filename']

    return image,width,label,length,text,filename

def _preprocess_image(image):
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5) #减法

    # Pad with copy of first row to expand to 32 pixels height
    first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
    # 从inputs中抽取部分内容
    #  inputs：可以是list,array,tensor
    #  begin：n维列表，begin[i] 表示从inputs中第i维抽取数据时，相对0的起始偏移量，也就是从第i维的begin[i]开始抽取数据
    #  size：n维列表，size[i]表示要抽取的第i维元素的数目
    image = tf.concat([first_row, image], 0)

    return image
    
