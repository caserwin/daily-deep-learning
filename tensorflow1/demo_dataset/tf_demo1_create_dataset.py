# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf
import numpy as np

b = {"a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
     "b": np.random.uniform(size=(5, 2))}

dataset = tf.data.Dataset.from_tensor_slices(b)

iterator = dataset.make_one_shot_iterator()

one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))
