#-*-coding:utf8-*-

__author ="buyizhiyou"
__date ="2018-7-26"

'''
generate data for training
'''
import sys,os,pdb
import numpy as np 
import random
import re

from PIL import Image
from PIL import ImageFont,ImageDraw,ImageFilter
from skimage.util  import random_noise
from skimage import io,transform

def get_len(line):
    '''
	return length of line,we regard one chinese char as 1 but one number and english char
	as 50% length compared with chinese char
	'''
    chinese_chars = re.findall(u'[\u4e00-\u9fa5]', line)  # chinese chars
    chinese_length = len(chinese_chars)  # length of chinese chars
    rest_leng = int(0.5*(len(line) - chinese_length)) # length of english chars,numbers and others

    length = chinese_length + rest_leng
    # print(length)

    return length


def gen_image(h,i,text,length):
    '''
    generate a sample accoding to text and length
    '''
    width = int(35*length+10)
    height = 40
   

    img = Image.new('RGB',(width,height),(255,255,255))
    draw = ImageDraw.Draw(img)
    imgs = []

    for k in range(1):
        fontsize = int(35+random.random())
        font = ImageFont.truetype('fonts/fz.ttf', fontsize)
        w0 = random.uniform(4,8)# align left
        h0 = (height - fontsize) // 2+random.uniform(0,1)  # start y
        draw.text((w0, h0), text, (0, 0, 0), font=font)
        img = np.array(img)
        r = random.uniform(0,1.5)
        #img_rotate = transform.rotate(img,r)
        imgs.append(img)
        #imgs.append(img_rotate)
        # #add gauss noise
        img_noise = random_noise(img,mode='gaussian')
        imgs.append(img_noise)
        # #rotate
        # r = random.uniform(0,1.5)
        # img_rotate = transform.rotate(img_noise,r)
        # imgs.append(img_rotate)
        # r = random.uniform(0,1.5)
        # img_rotate2 = transform.rotate(img_noise,-r)
        # imgs.append(img_rotate2)

        #add poisson noise
        img_noise2 = random_noise(img,mode='poisson')
        imgs.append(img_noise2)
        # #rotate
        # r = random.uniform(0,1.5)
        # img_rotate = transform.rotate(img_noise2,r)
        # imgs.append(img_rotate)
        # r = random.uniform(0,1.5)
        # img_rotate2 = transform.rotate(img_noise2,-r)
        # imgs.append(img_rotate2)

    for j in range(len(imgs)):
        io.imsave('sample/'+str(i)+"_"+str(j)+".jpg",imgs[j])
        h.write(str(i)+"_"+str(j)+".jpg"+' '+text+'\n')

    return len(imgs)


'''
generate numbers ,English characters and chinese characters;
you can uncomment it to generate train data;
`items.txt` is the file of lines you want to generate
'''
# with open('items.txt','r') as f:
#     lines = f.readlines()

# print("Begin Generating Data:")
# nums = 0
# for (i,line) in enumerate(lines):
#     length = get_len(line)
#     num = gen_image(i,line,length)
#     nums +=num
# print("Generate %d images!!!!"%nums)
# print("End Generating.")

'''
generate numbers and English characters 
'''
print("Begin Generating Data:")
nums = 0
vocab = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k',
                'l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

h = open('labels2.txt','w')
for i in range(10000):
    m = random.randint(4,8)
    text = ''.join(random.sample(vocab,m))#m个char,不定长
    print(text)
    num  = gen_image(h,i,text,m/2)
    nums +=num
h.close()
       
print("Generate %d images!!!!"%nums)
print("End Generating.")
