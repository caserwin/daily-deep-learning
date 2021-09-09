# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import (Zeros, glorot_normal)
from tensorflow.python.keras.layers import Layer


class CrossNet(Layer):
    def __init__(self, layer_num=2, l2_reg=0, seed=1024, **kwargs):
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        self.l2 = tf.contrib.layers.l2_regularizer(float(l2_reg))
        self.seed = seed
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = input_shape[-1].value
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(dim, 1),
                                        initializer=glorot_normal(seed=self.seed),
                                        regularizer=None,
                                        trainable=True) for i in range(self.layer_num)]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        if len(self.kernels) > 0:
            tf.contrib.layers.apply_regularization(self.l2, weights_list=self.kernels)

        # Be sure to call this somewhere!
        super(CrossNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
            dot_ = tf.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = tf.squeeze(x_l, axis=2)
        # todo 这里是debug 调试
        x_l = tf.Print(x_l, [x_l], message="test ", summarize=-1)
        sess.run(tf.global_variables_initializer())
        print(sess.run(x_l))

        return x_l

    def get_config(self, ):
        config = {'layer_num': self.layer_num, 'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super(CrossNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


sess = tf.InteractiveSession()
if __name__ == '__main__':
    fea1_embedding = tf.convert_to_tensor([[[1, 2]]], dtype=tf.float32)
    fea2_embedding = tf.convert_to_tensor([[[3, 4]]], dtype=tf.float32)
    fea3_embedding = tf.convert_to_tensor([[[5, 6]]], dtype=tf.float32)

    dense_value_list = [fea1_embedding, fea2_embedding, fea3_embedding]
    concat_fea = tf.keras.layers.Concatenate(axis=-1)(dense_value_list)

    dnn_input = tf.keras.layers.Flatten()(concat_fea)
    cross_out = CrossNet(2, l2_reg=0)(dnn_input)
