# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

# ================================== tf.argmax ======================================
logits = tf.constant([2, 20, 30, 3, 6])
predicted_classes = tf.argmax(logits)
with tf.Session() as sess:
    print(sess.run(predicted_classes))

logits = tf.constant([[2], [20], [30], [3], [6]], 1)
predicted_classes = tf.argmax(logits)
with tf.Session() as sess:
    print(sess.run(predicted_classes))

"""
返回索引
"""