# -*- coding: utf-8 -*-
# @Author : Erwin
# coding=utf-8
import tensorflow as tf
from tensorflow.python.keras.layers import Layer

fea1_embedding = tf.convert_to_tensor([[[1, 2]]], dtype=tf.float32)
fea2_embedding = tf.convert_to_tensor([[[3, 4]]], dtype=tf.float32)
fea3_embedding = tf.convert_to_tensor([[[5, 6]]], dtype=tf.float32)

sparse_emb_list = [fea1_embedding, fea2_embedding, fea3_embedding]

concated_embeds_value = tf.concat(sparse_emb_list, axis=1)


class FM(Layer):
    def call(self, inputs, **kwargs):
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)
        # TODO 可以直接打印变量调试。但是要事先执行 global_variables_initializer 的话，就不能通过这种方式打印变量
        print(sum_of_square.eval())
        return cross_term


# a = FM()(concated_embeds_value)  # 不能直接执行 sum_of_square.eval()，因为这段代码不在 sess 内。
with tf.Session() as session:
    # writer = tf.summary.FileWriter('/Users/casyd_xue/tf_demo3_tensorboard', session.graph)
    print(session.run(concated_embeds_value))
    print(session.run(FM()(concated_embeds_value)))  # 可以直接执行sum_of_square.eval()