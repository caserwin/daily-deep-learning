# -*- coding: utf-8 -*-
# @Author : Erwin
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from demo_mtl_loss.mtl_layer import CustomMultiLossLayer

np.random.seed(0)

N = 100
nb_epoch = 2000
batch_size = 20
nb_features = 1024
Q = 1
D1 = 1  # first output
D2 = 1  # second output


def gen_data(N):
    X = np.random.randn(N, Q)
    w1 = 2.
    b1 = 8.
    sigma1 = 1e1  # ground truth
    Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(N, D1)
    w2 = 3
    b2 = 3.
    sigma2 = 1e0  # ground truth
    Y2 = X.dot(w2) + b2 + sigma2 * np.random.randn(N, D2)
    return X, Y1, Y2


def get_prediction_model():
    inp = Input(shape=(Q,), name='inp')
    x = Dense(nb_features, activation='relu')(inp)
    y1_pred = Dense(D1)(x)
    y2_pred = Dense(D2)(x)
    return Model(inp, [y1_pred, y2_pred])


def get_trainable_model(prediction_model):
    inp = Input(shape=(Q,), name='inp')
    y1_pred, y2_pred = prediction_model(inp)
    y1_true = Input(shape=(D1,), name='y1_true')
    y2_true = Input(shape=(D2,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    return Model([inp, y1_true, y2_true], out)


X, Y1, Y2 = gen_data(N)
# pylab.figure(figsize=(3, 1.5))
# pylab.scatter(X[:, 0], Y1[:, 0])
# pylab.scatter(X[:, 0], Y2[:, 0])
# pylab.show()


prediction_model = get_prediction_model()
trainable_model = get_trainable_model(prediction_model)
trainable_model.compile(optimizer='adam', loss=None)
assert len(trainable_model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
assert len(trainable_model.losses) == 1

hist = trainable_model.fit([X, Y1, Y2], nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
# pylab.plot(hist.history['loss'])
ls = [np.exp(K.get_value(log_var[0])) ** 0.5 for log_var in trainable_model.layers[-1].log_vars]
for i in ls:
    print(i)
