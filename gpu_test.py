#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'


import tensorflow as tf
 
'''
with tf.device('/cpu:0'):
    a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')
    b = tf.constant([1.0,2.0,3.0],shape=[3],name='b')
with tf.device('/gpu:0'):
    c = a+b
   
#注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
#因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
print(sess.run(c))
'''


import tensorflow as tf
hello=tf.constant('hello,world')
sess=tf.Session()
print(sess.run(hello))

if __name__ == '__main__':
    pass
