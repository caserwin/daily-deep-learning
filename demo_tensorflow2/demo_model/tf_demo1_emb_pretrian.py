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
print(emb_model(test))

# ================================== method 2 ==================================
print("=" * 40, "method 2", "=" * 40)
tf.compat.v1.disable_eager_execution()
embedding_matrix = np.random.uniform(-1, 1, (vocab_size, embedding_dim))

W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
embedding_placeholder = tf.compat.v1.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

with tf.compat.v1.Session() as session:
    print(session.run(embedding_init, feed_dict={embedding_placeholder: embedding_matrix}))

"""
参考：
1. https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
2. https://stackoverflow.com/questions/62217537/tensorflow-keras-embedding-layer-applied-to-a-tensor
"""
# ================================== method 3 ==================================
print("=" * 40, "method 3", "=" * 40)
"""
运行以下代码时，不能开启：tf.compat.v1.disable_eager_execution()
参考：https://www.tensorflow.org/text/guide/word_embeddings
"""
embedding_layer = tf.keras.layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1, 2, 3]))
print(result)