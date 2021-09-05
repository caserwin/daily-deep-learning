# -*- coding: utf-8 -*-
# @Author : Erwin
# 定义输入、输出数据集
import tensorflow as tf
import numpy as np

X = tf.convert_to_tensor(np.random.normal(size=(10, 10)), dtype="float32")
Y = 3 * X + 2


# 定义模型表达
class MyModel():
    def __init__(self):
        self.W = tf.Variable(1., name='weight')
        self.B = tf.Variable(1., name='bias')

    def __call__(self, inputs):
        return inputs * self.W + self.B


# 定义损失函数
def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))


# 定义优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model = MyModel()

# 开始训练
for i in range(3000):
    with tf.GradientTape() as tape:
        Loss = loss(model, X, Y)
    grad = tape.gradient(Loss, [model.W, model.B])
    optimizer.apply_gradients(zip(grad, [model.W, model.B]))
    if i % 100 == 0:
        print(Loss)

# 训练结果
print(model.W.numpy(), model.B.numpy())
