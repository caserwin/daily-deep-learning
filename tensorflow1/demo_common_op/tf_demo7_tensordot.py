# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

x_l = tf.convert_to_tensor([[1, 2, 3, 4, 3, 2]], dtype=tf.float32)
kernels = tf.convert_to_tensor([[1], [1], [2], [2], [1], [1]], dtype=tf.float32)

xl_w = tf.tensordot(x_l, kernels, axes=(1, 0))
dot_ = tf.matmul(x_l, kernels)
with tf.Session() as session:
    print(session.run(xl_w))
    print(session.run(dot_))

"""
1. https://zhuanlan.zhihu.com/p/385099839
2. https://blog.csdn.net/sinat_36618660/article/details/100145804
"""