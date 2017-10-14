#-*-coding:utf8 -*-
import os
import natsort

filelist = os.listdir('./raw/')
filelist = natsort.natsorted(filelist)
g = open('annoations.txt','w')
for file in filelist:
	print(file[:-3])
	if file[-3:] =='txt':
		f= open('./raw/'+file,'r')
		text = f.read()
		path = './data/raw/'+file[:-3]+'bin.png'
		line = path+' '+text
		g.write(line)
		f.close()


g.close()


