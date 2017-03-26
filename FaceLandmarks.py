# -*- coding: utf-8 -*-

import dlib
import os
import numpy
import sys

from skimage import io

#源程序是用sys.argv从命令行参数去获取训练模型，精简版我直接把路径写在程序中了
predictor_path = "/home/administrator/dlib/dlib-18.16/shape_predictor_68_face_landmarks.dat"
#lujin huoqu
lujin = './presidents/'
for names in sys.argv[1:]:
  iiii=1
faces_path = lujin + names
#与人脸检测相同，使用dlib自带的frontal_face_detector作为人脸检测器
detector = dlib.get_frontal_face_detector()

#使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(predictor_path)


#使用skimage的io读取图片
img = io.imread(faces_path)


 #与人脸检测程序相同,使用detector进行人脸检测 dets为返回的结果
dets = detector(img, 1)


for k, d in enumerate(dets):

    #使用predictor进行人脸关键点识别 shape为返回的结果
    shape = predictor(img, d)


#也可以这样来获取（以一张脸的情况为例）
#get_landmarks()函数会将一个图像转化成numpy数组，并返回一个68 x2元素矩阵，输入图像的每个特征点对应每行的一个x，y坐标。
def get_landmarks(im):

    rects = detector(im, 1)

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

#多张脸使用的一个例子
def get_landmarks_m(im):

    dets = detector(im, 1)

    #脸的个数
    #print("Number of faces detected: {}".format(len(dets)))

    for i in range(len(dets)):

        facepoint = np.array([[p.x, p.y] for p in predictor(im, dets[i]).parts()])

        for i in range(68):

            #标记点
            im[facepoint[i][1]][facepoint[i][0]] = [232,28,8]        

    return im    



print(get_landmarks(img))

