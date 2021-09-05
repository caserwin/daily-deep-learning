# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

t = tf.constant([[123, 321]])  # 可以理解为 1 * 2 的矩阵
res1 = tf.squeeze(t)

with tf.Session() as session:
    print(session.run(res1))

print("=" * 80)

t2 = tf.constant([1, 2, 3, 4, 5, 6], shape=(2, 1, 3, 1))
with tf.Session() as session:
    print(session.run(tf.squeeze(t2, axis=1)))
    print(tf.squeeze(t2, axis=(1, 3)).shape)

"""
1. 该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果。
2. axis可以用来指定要删掉的为1的维度
"""
