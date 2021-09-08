# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

"""
1. https://blog.csdn.net/cc1949/article/details/78422704
"""

# 一维测试
a = tf.constant([1, 2, 3, 2])
with tf.Session() as sess:
    print("vector transpose:", sess.run(tf.transpose(a)))

print("=" * 80)
# 二维测试
a = tf.constant([1, 2, 3, 2], shape=[2, 2])

with tf.Session() as sess:
    print("vector transpose:", sess.run(tf.transpose(a, perm=[0, 1])))
    print("vector transpose:", sess.run(tf.transpose(a, perm=[1, 0])))
    print("vector transpose:", sess.run(tf.transpose(a)))

print("=" * 80)
# 三维测试
x = tf.constant([1, 2, 3, 4, 4, 3], shape=[1, 2, 3])

with tf.Session() as sess:
    a = tf.transpose(x, [0, 1, 2])  # [0, 1, 2] 保持原来顺序
    b = tf.transpose(x, [0, 2, 1])  # 转置第2个维度和第3个维度
    c = tf.transpose(x, [1, 0, 2])
    d = tf.transpose(x, [1, 2, 0])
    e = tf.transpose(x, [2, 1, 0])
    f = tf.transpose(x, [2, 0, 1])

    print(a)
    print(b)
    print(sess.run(b))
    print(sess.run(c))
