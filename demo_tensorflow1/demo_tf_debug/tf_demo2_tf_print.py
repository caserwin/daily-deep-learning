#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.constant([1.0, 3.0])
sess = tf.InteractiveSession()

a = tf.Print(a, [a], message="This is a: ")
b = tf.add(a, a)
print(b.eval())
