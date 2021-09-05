# -*- coding: utf-8 -*-
# @Author : Erwin
# coding=utf-8
"""
代码取自这里：https://github.com/yezhiwen/tensorflow_recommend_algorithm/blob/main/model/mmoe.py#L54
"""
import tensorflow as tf


class MMoE(tf.keras.layers.Layer):

    def __init__(self, units, num_experts, num_tasks,
                 use_expert_bias=True, use_gate_bias=True, expert_activation='relu', gate_activation='softmax',
                 expert_bias_initializer='zeros', gate_bias_initializer='zeros', expert_bias_regularizer=None,
                 gate_bias_regularizer=None, expert_bias_constraint=None, gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling', gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None, gate_kernel_regularizer=None, expert_kernel_constraint=None,
                 gate_kernel_constraint=None, activity_regularizer=None, **kwargs):
        super(MMoE, self).__init__(**kwargs)

        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = tf.keras.initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = tf.keras.initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = tf.keras.regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = tf.keras.regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = tf.keras.constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = tf.keras.constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = tf.keras.initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = tf.keras.initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = tf.keras.regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = tf.keras.regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = tf.keras.constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = tf.keras.constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []

        for i in range(self.num_experts):
            self.expert_layers.append(tf.keras.layers.Dense(self.units, activation=self.expert_activation,
                                                            use_bias=self.use_expert_bias,
                                                            kernel_initializer=self.expert_kernel_initializer,
                                                            bias_initializer=self.expert_bias_initializer,
                                                            kernel_regularizer=self.expert_kernel_regularizer,
                                                            bias_regularizer=self.expert_bias_regularizer,
                                                            activity_regularizer=None,
                                                            kernel_constraint=self.expert_kernel_constraint,
                                                            bias_constraint=self.expert_bias_constraint))
        for i in range(self.num_tasks):
            self.gate_layers.append(tf.keras.layers.Dense(self.num_experts, activation=self.gate_activation,
                                                          use_bias=self.use_gate_bias,
                                                          kernel_initializer=self.gate_kernel_initializer,
                                                          bias_initializer=self.gate_bias_initializer,
                                                          kernel_regularizer=self.gate_kernel_regularizer,
                                                          bias_regularizer=self.gate_bias_regularizer,
                                                          activity_regularizer=None,
                                                          kernel_constraint=self.gate_kernel_constraint,
                                                          bias_constraint=self.gate_bias_constraint))

    def call(self, inputs, **kwargs):

        expert_outputs, gate_outputs, final_outputs = [], [], []

        # inputs: (batch_size, embedding_size)
        for expert_layer in self.expert_layers:
            expert_output = tf.expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)

        # batch_size * units * num_experts
        expert_outputs = tf.concat(expert_outputs, 2)

        # [(batch_size, num_experts), ......]
        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            # (batch_size, 1, num_experts)
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)

            # (batch_size * units * num_experts) * (batch_size, 1 * units, num_experts)
            weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output, self.units,
                                                                                       axis=1)

            # (batch_size, units)
            final_outputs.append(tf.reduce_sum(weighted_expert_output, axis=2))

        # [(batch_size, units), ......]   size: num_task
        return final_outputs


if __name__ == '__main__':
    fea1_embedding = tf.convert_to_tensor([[[1, 2]]], dtype=tf.float32)
    fea2_embedding = tf.convert_to_tensor([[[3, 4]]], dtype=tf.float32)
    fea3_embedding = tf.convert_to_tensor([[[5, 6]]], dtype=tf.float32)

    dense_value_list = [fea1_embedding, fea2_embedding, fea3_embedding]
    concat_fea = tf.keras.layers.Concatenate(axis=-1)(dense_value_list)

    dnn_input = tf.keras.layers.Flatten()(concat_fea)
    mmoe_layers = MMoE(units=4, num_experts=8, num_tasks=2)(dnn_input)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print(session.run(mmoe_layers))