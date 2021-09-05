# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf
import numpy as np
import random

# ================================== tf.reshape demo1======================================
# 参考：https://zhuanlan.zhihu.com/p/51162209
x = tf.placeholder('float')
# y = tf.reshape(x, [-1, 4, 4, 1])
y = tf.reshape(x, [2, 16])

with tf.Session() as sess:
    x1 = np.asarray([random.uniform(0, 1) for i in range(32)])
    result = sess.run(y, feed_dict={x: x1})
    print(result)

print("=" * 80)
# ================================== tf.reshape demo2======================================
x = tf.constant([[1, 2], [3, 4]])
y = tf.reshape(x, [-1])
with tf.Session() as sess:
    print(sess.run(y))

print("=" * 80)
# ================================== tf.reshape demo2======================================
x = tf.ones((2, 2, 2))
y = x[:, tf.newaxis]
print(x)
print(y.shape)
