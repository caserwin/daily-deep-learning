# -*- coding: utf-8 -*-
# @Author : Erwin
import numpy as np
import tensorflow as tf

embedding_dim = 5
vocab_size = 100

# ================================== method 1 ==================================
print("=" * 40, "method 1", "=" * 40)
test = np.asarray([[1, 2, 3, 4], [4, 5, 6, 7]])

embedding_matrix = np.random.uniform(-1, 1, (vocab_size, embedding_dim))

emb_model = tf.keras.Sequential()
embedder = tf.keras.layers.Embedding(vocab_size,
                                     embedding_dim,
                                     trainable=False,
                                     weights=[embedding_matrix],
                                     input_shape=(None,))
emb_model.add(embedder)
res = emb_model(tf.convert_to_tensor(test))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(res))

# ================================== method 2 ==================================
print("=" * 40, "method 2", "=" * 40)
embedding_matrix = np.random.uniform(-1, 1, (vocab_size, embedding_dim))

W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

with tf.Session() as session:
    print(session.run(embedding_init, feed_dict={embedding_placeholder: embedding_matrix}))
"""
参考：
1. https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
2. https://stackoverflow.com/questions/62217537/tensorflow-keras-embedding-layer-applied-to-a-tensor
"""
# ================================== method 3 ==================================
embedding_layer = tf.keras.layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1, 2, 3]))
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(result))
"""
参考：https://www.tensorflow.org/text/guide/word_embeddings
"""