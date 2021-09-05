# -*- coding: utf-8 -*-
# @Author : Erwin
# 定义输入、输出数据集
import tensorflow as tf
import numpy as np

X = tf.convert_to_tensor(np.random.normal(size=(10, 10)), dtype="float32")
Y = 3 * tf.pow(X, 2) + 2 * X + 2 + np.random.normal(size=(10, 1))


# 定义模型表达
class MyModel():
    def __init__(self):
        self.W1 = tf.Variable(1., name='weight1')
        self.W2 = tf.Variable(1., name='weight2')
        self.B = tf.Variable(1., name='bias')

    def __call__(self, inputs):
        return tf.pow(inputs, 2) * self.W1 + inputs * self.W2 + self.B


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
    grad = tape.gradient(Loss, [model.W1, model.W2, model.B])
    optimizer.apply_gradients(zip(grad, [model.W1, model.W2, model.B]))
    if i % 100 == 0:
        print(Loss)

# 训练结果
print(model.W1.numpy(), model.W2.numpy(), model.B.numpy())
