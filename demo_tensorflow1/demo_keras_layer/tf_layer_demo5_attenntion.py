# -*- coding: utf-8 -*-
# @Author : Erwin
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from deepctr.layers.core import LocalActivationUnit

"""
代码取自 DeepCTR 
"""


class AttentionSequencePoolingLayer(Layer):
    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=False,
                 return_score=False,
                 supports_masking=False, att_name="", **kwargs):

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        self.att_name = att_name

        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        self.local_att = LocalActivationUnit(
            self.att_hidden_units, self.att_activation, l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, )
        super(AttentionSequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None, training=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            queries, keys = inputs
            key_masks = tf.expand_dims(mask[-1], axis=1)
        else:
            queries, keys, keys_length = inputs
            hist_len = keys.get_shape()[1]
            key_masks = tf.sequence_mask(keys_length, hist_len)

        attention_score = self.local_att([queries, keys], training=training)
        # print("==", attention_score)
        # attention_score = tf.Print(attention_score, [attention_score], message="this is attention_score: ")
        # print("--", attention_score)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print(session.run(attention_score))

        outputs = tf.transpose(attention_score, (0, 2, 1))
        outputs = tf.Print(outputs, [outputs], message="this is attention_score: ")

        if self.weight_normalization:
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)

        outputs = tf.where(key_masks, outputs, paddings)
        if self.weight_normalization:
            outputs = tf.nn.softmax(outputs)

        if not self.return_score:
            outputs = tf.matmul(outputs, keys)

        outputs._uses_learning_phase = attention_score._uses_learning_phase
        return outputs

    def compute_output_shape(self, input_shape):
        if self.return_score:
            return (None, 1, input_shape[1][1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'return_score': self.return_score,
                  'supports_masking': self.supports_masking}
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    embedding_layer = tf.keras.layers.Embedding(input_dim=1000,
                                                output_dim=5,
                                                mask_zero=True,
                                                embeddings_initializer=tf.keras.initializers.Ones())
    keys_emb = embedding_layer(tf.constant([[1, 2, 3]]))

    query_emb = tf.convert_to_tensor([[[1, 1, 1, 2, 2]]], dtype=tf.float32)

    hist = AttentionSequencePoolingLayer((80, 40), "dice", weight_normalization=False, supports_masking=True)(
        [query_emb, keys_emb])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print("=============== keys ===============")
        print(session.run(keys_emb))
        print("=============== attention vector ===============")
        print(session.run(hist))
