# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf
import numpy as np

t = tf.constant([[123, 321]])  # 可以理解为 1 * 2 的矩阵
print(t.shape)
print("=" * 80)
res1 = tf.expand_dims(t, axis=0)  # 可以理解为 1 * 1 * 2 的矩阵
res2 = tf.expand_dims(t, axis=1)  # 可以理解为 1 * 1 * 2 的矩阵
res3 = tf.expand_dims(t, axis=2)  # 可以理解为 1 * 2 * 1 的矩阵

with tf.Session() as session:
    print(session.run(res1))
    print(session.run(res2))
    print(session.run(res3))

print("=" * 80)

t2 = np.zeros((2, 3, 5))
print(t2.shape)
t3 = tf.expand_dims(t2, 0)
t4 = tf.expand_dims(t2, 2)
t5 = tf.expand_dims(t2, 3)
print(t3.shape)
print(t4.shape)
print(t5.shape)
"""
1. 如果axis是负数，则从最后向前索引。
"""