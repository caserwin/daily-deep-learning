# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf
import numpy as np

# 1000 个样本，每个样本 20 维特征，就是 (1000, 20)。
x_train = np.random.random((1000, 20))
# 1000 个样本，每个样本有 0-9 十个类别，做了one-hot之后就是 (1000, 10)。
y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

x_test = np.random.random((100, 20))
y_test = tf.keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

print(x_test)
print(y_test)