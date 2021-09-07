#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# 必须开启 enable_eager_execution
tf.enable_eager_execution()
x = tf.constant([2, 3, 4, 5])
y = tf.constant([20, 30, 40, 50])
z = tf.add(x, y)

print(z)
tf.print(x, output_stream=sys.stdout)
tf.print(z, output_stream=sys.stdout)
