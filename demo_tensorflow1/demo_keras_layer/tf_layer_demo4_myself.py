# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

"""
https://keras.io/zh/layers/writing-your-own-keras-layers/
"""


class MyLayer(tf.keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1].value, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.tensordot(x, self.kernel, axes=(1, 0))
        # return tf.tensordot(x, self.kernel, axes=1)    # 和上面结果是一样的

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


if __name__ == '__main__':
    dnn_input = tf.convert_to_tensor([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    my_layers = MyLayer(output_dim=4)(dnn_input)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(my_layers))
