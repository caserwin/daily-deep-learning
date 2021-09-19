# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf
import numpy as np

"""
本质上选取一个张量里面索引对应的元素
"""
# 创建一个张量
c = np.random.random([10, 2, 2])
# 对张量c 取索引列
b = tf.nn.embedding_lookup(c, [1, 3])

with tf.Session() as sess:
    print(c)
    print("==" * 40)
    print(sess.run(b))