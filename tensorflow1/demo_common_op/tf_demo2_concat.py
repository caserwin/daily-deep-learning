# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

fea1_embedding = tf.convert_to_tensor([[[1, 2]]], dtype=tf.float32)
fea2_embedding = tf.convert_to_tensor([[[3, 4]]], dtype=tf.float32)
fea3_embedding = tf.convert_to_tensor([[[5, 6]]], dtype=tf.float32)

emb_list = [fea1_embedding, fea2_embedding, fea3_embedding]

with tf.Session() as session:
    print(session.run(tf.concat(emb_list, axis=0)))
    print(session.run(tf.concat(emb_list, axis=1)))
    print(session.run(tf.concat(emb_list, axis=2)))

print("=" * 80)

fea4_embedding = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
fea5_embedding = tf.convert_to_tensor([[7, 8, 9], [10, 11, 12]], dtype=tf.float32)

emb_list = [fea4_embedding, fea5_embedding]
with tf.Session() as session:
    print(session.run(tf.concat(emb_list, 0)))
    print(session.run(tf.concat(emb_list, 1)))

print("=" * 80)
fea6_embedding = tf.convert_to_tensor([[[1, 2], [3, 4]]], dtype=tf.float32)
fea7_embedding = tf.convert_to_tensor([[[3, 4], [5, 6]]], dtype=tf.float32)
emb_list = [fea6_embedding, fea7_embedding]
with tf.Session() as session:
    print(session.run(tf.concat(emb_list, 0)))  # 2*2*2 矩阵
    print(session.run(tf.concat(emb_list, 1)))  # 1*4*2 矩阵
    print(session.run(tf.concat(emb_list, 2)))  # 1*2*4 矩阵

"""
说明：
两个tensor :
a 是 (1, 1, 2)
b 是 (1, 1, 2)

如果按照 axis = 0 concat 就是(2, 1, 2) 的tensor。
如果按照 axis = 1 concat 就是(1, 2, 2) 的tensor。
如果按照 axis = 2 concat 就是(1, 1, 4) 的tensor。
"""