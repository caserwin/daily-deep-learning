# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

# ================================== sigmoid demo======================================
"""
记录两点：
1. dtype=tf.float32 不写会遇到侧错，解决参考：https://stackoverflow.com/questions/44417133/typeerror-value-passed-to-parameter-a-has-datatype-not-in-list-of-allowed-val/44417286
2. how to convert logits to probability in binary classification in tensorflow?：https://stackoverflow.com/questions/46416984/how-to-convert-logits-to-probability-in-binary-classification-in-tensorflow
"""
logit = tf.constant([-0.1, 2, 3, 4], dtype=tf.float32)
# prediction = tf.round(tf.nn.sigmoid(logit))
prediction_softmax = tf.nn.softmax(logit)
prediction_sigmoid = tf.nn.softmax(logit)
with tf.Session() as sess:
    print(sess.run(prediction_softmax))
    print(sess.run(prediction_sigmoid))
    print(sess.run(tf.round(tf.nn.sigmoid(logit))))
    print(sess.run(tf.nn.sigmoid(logit)))
    print(sess.run(tf.identity(tf.nn.sigmoid(logit), name="rank_predict")))
