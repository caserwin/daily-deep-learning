# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

# ================================== tf.string demo======================================
value = ["a,b"]
y = tf.string_split(value, delimiter=',')
with tf.Session() as sess:
    print(sess.run(y))

# ================================== tf.string demo======================================
"""
如果开启 enable_eager_execution 则上面代码要注销掉
"""
tf.enable_eager_execution()
print (tf.string_split(['this is example sentence']).values)
