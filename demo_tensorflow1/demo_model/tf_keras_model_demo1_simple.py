# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf
from tensorflow.python.keras.models import Model

"""
参考：https://stackoverflow.com/questions/51672903/keras-tensorflow-how-to-set-breakpoint-debug-in-custom-layer-when-evaluating
"""


def my_init(shape, dtype=None):
    # return tf.keras.backend.random_normal(shape, dtype=dtype)
    return tf.keras.backend.ones(shape, dtype=dtype)


class SimpleModel(Model):
    """
    这里 relu 和 tf.keras.backend.relu 这种写法都可以的。
    """

    def __init__(self):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(2, activation="relu", kernel_initializer="glorot_uniform")
        self.dense1 = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform")

    def call(self, inputs):
        z = self.dense0(inputs)
        z = self.dense1(z)
        return z


x = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

model0 = SimpleModel()
y0 = model0.call(x)
print(y0)

model1 = SimpleModel()
model1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy)
y1 = model1.predict(x, steps=1)
print(y1)
