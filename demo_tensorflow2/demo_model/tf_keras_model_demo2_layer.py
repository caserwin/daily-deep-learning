# -*- coding: utf-8 -*-
# @Author : Erwin
# coding=utf-8
import tensorflow as tf


def my_init(shape, dtype=None):
    # return tf.keras.backend.random_normal(shape, dtype=dtype)
    return tf.keras.backend.ones(shape, dtype=dtype)


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, output_dim, job_type, if_bias=False, **kwargs):
        self.output_dim = output_dim
        self.job_type = job_type
        self.if_bias = if_bias
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.job_type == 'myself':
            self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape[-1], self.output_dim),
                                          initializer=tf.keras.initializers.Constant(
                                              my_init((input_shape[-1], self.output_dim), )),
                                          trainable=True)
        else:
            self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape[-1], self.output_dim),
                                          initializer='uniform',
                                          trainable=True)

        if self.if_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(input_shape[0], self.output_dim),
                                        initializer=tf.keras.initializers.Ones(),
                                        trainable=True)
        else:
            self.bias = self.add_weight(name='bias',
                                        shape=(input_shape[0], self.output_dim),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)

        super(MyLayer, self).build(input_shape)

    def call(self, x):
        return tf.tensordot(x, self.kernel, axes=(1, 0)) + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


if __name__ == '__main__':
    dnn_input = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    print(MyLayer(output_dim=4, job_type='myself')(dnn_input))
    print("=" * 80)
    print(MyLayer(output_dim=4, job_type='myself', if_bias=True)(dnn_input))
    print("=" * 80)
    print(MyLayer(output_dim=4, job_type='uniform')(dnn_input))

"""
1. constant 和 convert_to_tensor 效果是一样的。
2. eager 执行，可以直接输出。
"""