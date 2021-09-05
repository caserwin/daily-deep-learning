#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/5 11:33 上午
# @Author : Erwin
import tensorflow as tf
import numpy as np


def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg


print(my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]])))
print(my_func([[1.0, 2.0], [3.0, 4.0]]))
print(my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)))
with tf.Session() as session:
    print(session.run(my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))))  # 2*2*2 矩阵
    print(session.run(my_func([[1.0, 2.0], [3.0, 4.0]])))  # 1*4*2 矩阵
    print(session.run(my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))))  # 1*2*4 矩阵
