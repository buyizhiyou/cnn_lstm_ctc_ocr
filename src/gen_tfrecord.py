#-*-coding:utf8 -*-

import os
import tensorflow as tf
import math
import pdb

"""Each record within the TFRecord file is a serialized Example proto. 
The Example proto contains the following fields:
  image/encoded: string containing JPEG encoded grayscale image
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/filename: string containing the basename of the image file
  image/labels: list containing the sequence labels for the image text
  image/text: string specifying the human-readable version of the text
"""

#out_charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
out_charset = "单位NUL/hpμlngmoSceuPd%10E9f2秒 FiDCOIHskM小时或℃A4拷贝个G-万K5WB3月日.867男：女<尿天随机~:≥绝经前—后;+正常饮食低钠卵泡期黄体未怀孕排X岁*^少见检测项目抗梅毒螺旋特异性（T)丙型肝炎(tV乙表面原人免疫缺陷病颜色状蛔虫钩鞭吞噬细胞油滴夏雷登氏脓霉菌孢子红白不消化物隐血试验R有核清晰度外观潘肌钙蛋亚硝酸盐质酮浊胆素葡萄糖管类酵母粘液丝结晶比重上皮电导率值脂酶甲胎铁r癌胚链定心极密固醇）甘三酯无磷高球腺苷脱氨氯a门冬基转移总α岩藻直接b间钾汁氮碱γ谷酰脯二肽酐脑脊反应激同工活羟丁氢乳嗜粒板压积淋巴平均分布宽计数中浓含量国际标准凝对照聚纤维部超敏胱抑y肾滤过全实剩余吸入氧流碳根温纠阴离隙饱和渗透镁碘游促说明沉降淀粉引Y环瓜风湿因溶双J着点列β网织百端利急诊李凡他草鳞相关神元烯角片段枝杆胸水补视合微Ⅰ周本颗快速力旁内载半镜纳紧张浆醛Ⅱ生成胺餐巨腹胰岛绒毛膜线组增殖纺锤动波形尔髓备注析底戊庚非轻λ药克莫司胶殊序骨延长辅助制共读,群空羧露真稀释其糜集，可肠歧酪梭普拉柔嫩感染论呼气钟果帕卧隆κ泌雌刺滑睾幼稚叶层地辛鲎放例酚连ⅣⅢ鉴铜蓝肺支一灰石q幽现症服附纯疱疹弓综出热多去肪的术四粪便规寄、查[房]肿瘤志＋系解谱五功能法输套自费脉泳传研静科两辩呕吐及代谢会荧光室植用残留灶式涂学道种优势身簇所带培养－栓弹图份杂交行儿茶弱阳软淡棕于限褐>黑糊烂绿块浑建议复详亮暗澈±硬干扰严文报告腔仅供参考米|"


jpeg_data = tf.placeholder(dtype=tf.string)
jpeg_decoder = tf.image.decode_jpeg(jpeg_data,channels=1)

kernel_sizes = [5,5,3,3,3,3] # CNN kernels for image reduction

# Minimum allowable width of image after CNN processing
min_width = 20

def calc_seq_len(image_width):
    """Calculate sequence length of given image after CNN processing"""
    
    conv1_trim =  2 * (kernel_sizes[0] // 2)
    fc6_trim = 2*(kernel_sizes[5] // 2)
    
    after_conv1 = image_width - conv1_trim 
    after_pool1 = after_conv1 // 2
    after_pool2 = after_pool1 // 2
    after_pool4 = after_pool2 - 1 # max without stride
    after_fc6 =  after_pool4 - fc6_trim
    seq_len = 2*after_fc6
    return seq_len

seq_lens = [calc_seq_len(w) for w in range(1200)]

def gen_data(input_base_dir, image_list_filename, output_filebase, 
             num_shards=1000,start_shard=0):
    """ Generate several shards worth of TFRecord data """
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    sess = tf.Session(config=session_config)
    image_filenames = get_image_filenames(os.path.join(input_base_dir,
                                                      image_list_filename))
    #shard : 碎片　-->batch?
    num_digits = math.ceil( math.log10( num_shards - 1 ))
    shard_format = '%0'+ ('%d'%num_digits) + 'd' # Use appropriate # leading zeros
    images_per_shard = int(math.ceil( len(image_filenames) / float(num_shards) ))
    #pdb.set_trace()
    for i in range(start_shard,num_shards):
        start = i*images_per_shard
        end   = (i+1)*images_per_shard
        out_filename = output_filebase+'-'+(shard_format % i)+'.tfrecord'
        if os.path.isfile(out_filename): # Don't recreate data if restarting
            continue
        print (str(i),'of',str(num_shards),'[',str(start),':',str(end),']',out_filename)
        gen_shard(sess, input_base_dir, image_filenames[start:end], out_filename)
    # Clean up writing last shard
    start = num_shards*images_per_shard
    out_filename = output_filebase+'-'+(shard_format % num_shards)+'.tfrecord'
    print (str(i),'of',str(num_shards),'[',str(start),':]',out_filename)
    gen_shard(sess, input_base_dir, image_filenames[start:], out_filename)

    sess.close()

def gen_shard(sess, input_base_dir, image_filenames, output_filename):
    """Create a TFRecord file from a list of image filenames"""
    writer = tf.python_io.TFRecordWriter(output_filename)
    
    for filename in image_filenames:
        path_filename = os.path.join(input_base_dir,filename)
        if os.stat(path_filename).st_size == 0:#stat 用来返回相关文件的系统状态信息的
            print('SKIPPING',filename)
            continue
        try:
            image_data,height,width = get_image(sess,path_filename)#获取图像数据信息
            #pdb.set_trace()
            text,labels = get_text_and_labels(filename)#获取图像的text和label
            if is_writable(width,text):#判断图片和文本是否符合要求
                example = make_example(filename, image_data, labels, text, 
                                       height, width)#将每张图片的信息存到tfrecord的example中
                writer.write(example.SerializeToString())
            else:
                print('SKIPPING',filename)
        except:
            # Some files have bogus payloads, catch and note the error, moving on
            print('ERROR',filename)
    writer.close()


def get_image_filenames(image_list_filename):
    """ Given input file, generate a list of relative filenames"""
    filenames = []
    with open(image_list_filename) as f:
        for line in f:
            filename = line.split(' ',1)[0][:] # split off "./" and number
            filenames.append(filename)
    return filenames

def get_image(sess,filename):
    """Given path to an image file, load its data and size"""
    #pdb.set_trace()
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
    image = sess.run(jpeg_decoder,feed_dict={jpeg_data: image_data})
    height = image.shape[0]
    width = image.shape[1]
    return image_data, height, width

def is_writable(image_width,text):
    """Determine whether the CNN-processed image is longer than the string"""
    #return (image_width > min_width) and (len(text) <= seq_lens[image_width])
    return (image_width > min_width) and (len(text) <= 10)
    
def get_text_and_labels(filename):
    """ Extract the human-readable text and label sequence from image filename"""
    with open("../data/txt/"+filename[:-7]+'txt','r') as f:
        text = f.read()[:-1]
    # Transform string text to sequence of indices using charset, e.g.,
    # MONIKER -> [12, 14, 13, 8, 10, 4, 17]
    labels = [out_charset.index(c) for c in list(text)]#字符转为数值
    return text,labels

def make_example(filename, image_data, labels, text, height, width):
    """Build an Example proto for an example.
    Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_data: string, JPEG encoding of grayscale image
    labels: integer list, identifiers for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
    example = tf.train.Example(features=tf.train.Features(feature={  #tf.train.Feature的作用是指定数据格式转换
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
        'image/labels': _int64_feature(labels),
        'image/height': _int64_feature([height]),
        'image/width': _int64_feature([width]),
        'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'text/string': _bytes_feature(tf.compat.as_bytes(text)),
        'text/length': _int64_feature([len(text)])
    }))
    return example

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def main(argv=None):
    
    gen_data('../data/raw', 'annotations_train.txt', '../data/train/words')
    gen_data('../data/raw', 'annotations_val.txt',   '../data/val/words')
    gen_data('../data/raw', 'annotations_test.txt',  '../data/test/words')

if __name__ == '__main__':
    main()
