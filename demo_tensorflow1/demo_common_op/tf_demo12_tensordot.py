# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

# 一维测试
a = tf.constant([1, 2, 3, 2])
b = tf.constant([4, 5, 4, 1])
with tf.Session() as sess:
    print("outer product size:", tf.tensordot(a, b, axes=0))
    print("inner product size:", tf.tensordot(a, b, axes=1))
    print("outer product:", sess.run(tf.tensordot(a, b, axes=0)))
    print("inner product:", sess.run(tf.tensordot(a, b, axes=1)))

print("=" * 80)
# 二维测试
a = tf.constant([1, 2, 3, 2], shape=[2, 2])
b = tf.constant([4, 5, 4, 1], shape=[2, 2])

with tf.Session() as sess:
    print("outer product size:", tf.tensordot(a, b, axes=0))
    print("inner product size:", tf.tensordot(a, b, axes=1))
    print("outer product:", sess.run(tf.tensordot(a, b, axes=0)))
    print("inner product:", sess.run(tf.tensordot(a, b, axes=1)))

print("=" * 80)
# 三维测试
a = tf.constant([1, 2, 3, 4, 4, 3, 2, 1], shape=[2, 2, 2])
b = tf.constant([4, 5], shape=[2, 1])

with tf.Session() as sess:
    print("res_value:", sess.run(tf.tensordot(a, b, axes=(2, 0))))
    print("res_value:", sess.run(tf.tensordot(a, b, axes=1)))

"""
1. 有一个规律：a的下标接着b的下标，去掉axes所选取的下标，也就是 “ i j k l m i ” 去掉所有 " i " ，就是" j k l m "。
2. tf.tensordot(a, b, axes=1)      表示：a 取2，b取0。
3. tf.tensordot(a, b, axes=(0, 1)) 表示：a 取0，b取1。
"""
