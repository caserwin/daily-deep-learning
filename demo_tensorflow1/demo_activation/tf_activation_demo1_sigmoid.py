# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf
import math

a = tf.constant([1, 2, 3, 2], shape=[1, 4], dtype=float)
with tf.Session() as sess:
    print('softmax :', sess.run(tf.nn.softmax(a)))

softmax_out = tf.keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='ones', bias_initializer='zeros')(
    a)
sigmoid_out = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid, kernel_initializer='ones', bias_initializer='zeros')(
    a)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(softmax_out))
    print("=" * 40)
    print(sess.run(sigmoid_out))
    print(1 / (1 + math.exp(-8)))