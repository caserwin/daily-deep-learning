# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

res1 = tf.convert_to_tensor([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=tf.float32)  # shape 3*1*2
res2 = tf.convert_to_tensor([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)  # shape 3*2
res3 = tf.convert_to_tensor([[[1, 1], [2, 2]], [[1, 2], [1, 3]], [[2, 3], [3, 4]]], dtype=tf.float32)  # shape 3*2*2

dnn_input1 = tf.keras.layers.Flatten()(res1)
dnn_input2 = tf.keras.layers.Flatten()(res2)
dnn_input3 = tf.keras.layers.Flatten()(res3)

with tf.Session() as session:
    print(session.run(dnn_input1))
    print(session.run(dnn_input2))
    print(session.run(dnn_input3))

"""
1. Flatten: 在保留第0轴的情况下对输入的张量进行Flatten(扁平化)
2. 以下三种写法是一致的：
    dnn_input1 = tf.keras.layers.Flatten()(res1)
    dnn_input2 = tf.layers.Flatten()(res1)
    dnn_input3 = tf.layers.flatten(res1)
"""