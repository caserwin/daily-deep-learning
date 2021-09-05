# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf

fea1_embedding = tf.convert_to_tensor([[[1, 2]]], dtype=tf.float32)
fea2_embedding = tf.convert_to_tensor([[[3, 4]]], dtype=tf.float32)
fea3_embedding = tf.convert_to_tensor([[[5, 6]]], dtype=tf.float32)

emb_list = [fea1_embedding, fea2_embedding, fea3_embedding]

with tf.Session() as session:
    res0 = tf.concat(emb_list, axis=0)
    res1 = tf.concat(emb_list, axis=1)
    res2 = tf.concat(emb_list, axis=2)

    print(session.run(res0))
    print(session.run(res1))
    print(session.run(res2))
    print("=" * 80)
    print(session.run(tf.reduce_sum(res0, axis=0, keep_dims=True)))
    print(session.run(tf.reduce_sum(res0, axis=1, keep_dims=True)))
    print(session.run(tf.reduce_sum(res0, axis=2, keep_dims=True)))