#-*-coding:utf8 -*-
import os
import cv2
import pdb
import numpy as np 
import natsort

files = os.listdir('./raw/')

for file in files:
	if file[-3:]=='png':
		os.rename('./raw/'+file,'./raw/'+file[:-7]+'png')
#pdb.set_trace()
files = os.listdir('./raw/')
for file in files:
	if file[-3:]=='png':
		img=cv2.imread('./raw/'+file)
		shape = img.shape
		gimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
		ret=cv2.adaptiveThreshold(gimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		cv2.imwrite('./raw/'+file[:-3]+'bin.png',ret)
		os.remove('./raw/'+file)
#pdb.set_trace()
files = os.listdir("./raw/")
for file in files:
	if file[-7:]=='bin.png':
		im = cv2.imread('./raw/'+file,-1)
		im = cv2.resize(im,(256,64))
		cv2.imwrite('./raw/'+file,im)




