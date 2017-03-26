# -*- coding: utf-8 -*-
import os
import imghdr

#rename del
print('[1]Delete bad img , rename img')
reimgnum=0
delimg=0
zpname=[]
for zpname1 in os.listdir(r'./presidents'):	
	img='./presidents/'+zpname1
	imgType = imghdr.what(img)
	testtype = 'jpeg'   
	if imgType != testtype:
	    os.remove(img) 
            print('DELETE:'+img)
            delimg=delimg + 1
	      
	else:
	  reimgnum=reimgnum + 1
          reimg ='./presidents/'+bytes(reimgnum)+'.jpg'
	  #print(reimg)
          os.rename(img,reimg)	
print('ok,del:'+bytes(delimg)+'  rename:'+bytes(reimgnum))

#dlib python pjl1.py

#imgnumber
imgnumber = len(os.listdir('./presidents'))
print(imgnumber)
#
print('[2]shape predictor 68 face landmarks in text')
worknum=0
zp=[]
for zp1 in os.listdir(r'./presidents'):
	zp.append(zp1)

a = 'python FaceLandmarks.py '
filesave=[]
for ok in zp :
	c1 = '>>./presidents/'
	c2='.txt'
	c=c1+ok+c2   #>>xxx.jpg.txt
	b = a + ok + c 
        worknum = worknum + 1
        print('('+bytes(worknum)+'/'+bytes(imgnumber)+')')
	os.system(b)

#python FaceLandmarks.py xxx.jpg >>xxx.jpg.txt
