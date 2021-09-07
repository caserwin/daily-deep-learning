# -*- coding: utf-8 -*-
# @Author : Erwin
import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, [1])


def body(x):
    a = tf.constant(np.array([2]), dtype=tf.float32)
    x = a + x
    x = tf.Print(x, [x], message="test ")
    return x


def condition(x):
    return tf.reduce_sum(x) < 10


with tf.Session() as sess:
    result = tf.while_loop(cond=condition, body=body, loop_vars=[x])
    result_out = sess.run([result], feed_dict={x: np.zeros(1)})
    print(result_out)
