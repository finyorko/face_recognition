# coding:utf-8
from skimage import io, transform
#import glob
#import os
import tensorflow as tf
#import numpy as np
#import time
import cv2
import dlib
#import sys

detector = dlib.get_frontal_face_detector ( )  # 获取人脸分类器

ID = (1511346,1610731,1610763,1610260,1611407,1611408,
      1611409,1611412,1611413,1611415,1611417,1611418,
      1611419,1611420,1611421,1611424,1611425,1611426,
      1611427,1611430,1611431,1611433,1611434,1611436,
      1611437,1611438,1611440,1611444,1611446,1611447,
      1611449,1611450,1611451,1511453,1611455,1611458,
      1611459,1611460,1611461,1611462,1611470,1611471,
      1611472,1611472,1611476,1611478,1611480,1611482,
      1611483,1611486,1611487,1611488,1611490,1611491,
      1611492,1611493,1611494,1613371,1613376,1613378,
      1613550,1711459
      )

w = 128
h = 128
c = 3

x = tf.placeholder ( tf.float32, shape=[None, w, h, c], name='x' )
y_ = tf.placeholder ( tf.int32, shape=[None, ], name='y_' )


def CNNlayer():
    # 第一个卷积层（128——>64)
    conv1 = tf.layers.conv2d (
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer ( stddev=0.01 ) )
    pool1 = tf.layers.max_pooling2d ( inputs=conv1, pool_size=[2, 2], strides=2 )

    # 第二个卷积层(64->32)
    conv2 = tf.layers.conv2d (
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer ( stddev=0.01 ) )
    pool2 = tf.layers.max_pooling2d ( inputs=conv2, pool_size=[2, 2], strides=2 )

    # 第三个卷积层(32->16)
    conv3 = tf.layers.conv2d (
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer ( stddev=0.01 ) )
    pool3 = tf.layers.max_pooling2d ( inputs=conv3, pool_size=[2, 2], strides=2 )

    # 第四个卷积层(16->8)
    conv4 = tf.layers.conv2d (
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer ( stddev=0.01 ) )
    pool4 = tf.layers.max_pooling2d ( inputs=conv4, pool_size=[2, 2], strides=2 )

    re1 = tf.reshape ( pool4, [-1, 8 * 8 * 128] )

    # 全连接层
    dense1 = tf.layers.dense ( inputs=re1,
                               units=1024,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer ( stddev=0.01 ),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer ( 0.003 ) )
    dense2 = tf.layers.dense ( inputs=dense1,
                               units=512,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer ( stddev=0.01 ),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer ( 0.003 ) )
    logits = tf.layers.dense ( inputs=dense2,
                               units=68,
                               activation=None,
                               kernel_initializer=tf.truncated_normal_initializer ( stddev=0.01 ),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer ( 0.003 ) )
    return logits


# ---------------------------网络结束---------------------------
logits = CNNlayer ( )
predict = tf.argmax ( logits, 1 )

saver = tf.train.Saver ( )
sess = tf.Session ( )
saver.restore ( sess, 'ckpt1/faces.ckpt-9' )

user = input ( "图片（G）还是摄像头（V）:" )
if user == "G":
    path = input ( "图片路径名是：" )
    img = cv2.imread ( path )
    dets = detector ( img, 1 )
    print ( "Number of faces detected: {}".format ( len ( dets ) ) )
    for index, face in enumerate ( dets ):
        print (
            'face {}; left {}; top {}; right {}; bottom {}'.format ( index, face.left ( ), face.top ( ), face.right ( ),
                                                                     face.bottom ( ) ) )
        left = face.left ( )
        top = face.top ( )
        right = face.right ( )
        bottom = face.bottom ( )
        cv2.rectangle ( img, (left, top), (right, bottom), (0, 255, 0), 3 )
        io.imsave ( 'temp.png', img )
        img1 = io.imread ( 'temp.png' )
        img1 = transform.resize ( img1, (w, h, c) )
        cv2.imshow ( 'image', img1 )

        img1 = img[top:bottom, left:right]
        img1 = transform.resize ( img1, (w, h, c) )
        # cv2.imshow('image1',img)
        res = sess.run ( predict, feed_dict={x: [img1]} )
        print ( ID[res[0]] )
    if len ( dets ) == 0:
        img = transform.resize ( img, (w, h, c) )
        res = sess.run ( predict, feed_dict={x: [img]} )
        print ( ID[res[0]] )
        cv2.waitKey ( 0 )
        cv2.destroyAllWindows ( )
    cv2.waitKey ( 0 )
    cv2.destroyAllWindows ( )
else:
    # 打开摄像头
    cap = cv2.VideoCapture ( 0 )
    # 视屏封装格式

    while True:
        ret, frame = cap.read ( )
        gray = cv2.cvtColor ( frame, cv2.COLOR_BGR2GRAY )
        cv2.imshow ( 'frame', frame )

        # 抓取图像
        if cv2.waitKey ( 1 ) & 0xFF == ord ( 's' ):
            cv2.imwrite ( 'now.png', frame )

            img = cv2.imread ( "now.png" )
            dets = detector ( img, 1 )
            print ( "Number of faces detected: {}".format ( len ( dets ) ) )
            for index, face in enumerate ( dets ):
                print ( 'face {}; left {}; top {}; right {}; bottom {}'.format ( index,
                                                                                 face.left ( ), face.top ( ),
                                                                                 face.right ( ), face.bottom ( ) ) )
                left = face.left ( )
                top = face.top ( )
                right = face.right ( )
                bottom = face.bottom ( )
                img = img[top:bottom, left:right]

            # img=io.imread('image/now.png')
            img = transform.resize ( img, (w, h, c) )
            res = sess.run ( predict, feed_dict={x: [img]} )
            print ( ID[res[0]] )

        # 退出
        if cv2.waitKey ( 1 ) & 0xFF == ord ( 'q' ):
            break

    cap.release ( )
    cv2.destroyAllWindows ( )


























