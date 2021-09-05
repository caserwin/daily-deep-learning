# -*- coding: utf-8 -*-
# @Author : Erwin
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np

"""
读数据存为 numpy.ndarray 类型，再通过 from_tensor_slices 转成TF 可训练的类型。
"""

num_features = 784
batch_size = 64

# ======================step 1===========================
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# ======================step 2===========================
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(10000).batch(batch_size).prefetch(1)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_data = test_data.repeat().batch(batch_size).prefetch(1)

print(train_data)
print(test_data)
