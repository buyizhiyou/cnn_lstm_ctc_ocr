#-*-coding:utf8-*-

__author__ = "buyizhiyou"
__date__ = "2017-11-8"

import os
import random
import shutil

'''
对不同的长度序列生成对应的txt,方便训练时数据读取
'''
shutil.rmtree('txt')
os.mkdir('txt')
with open('labels2.txt','r') as f:
	lines = f.readlines()
	random.shuffle(lines)
	train_lines = lines[:int(0.9*len(lines))]
	val_lines = lines[int(0.9*len(lines)):]
	for line in train_lines:
		text = line.split(' ')[1]
		with open('txt/train_'+str(len(text)-1)+'.txt','a') as g:
			g.write(line)
	for line in val_lines:
		text = line.split(' ')[1]
		with open('txt/val_'+str(len(text)-1)+'.txt','a') as h:
			h.write(line)

