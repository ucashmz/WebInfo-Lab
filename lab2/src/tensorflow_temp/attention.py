import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras


class WordAttention(keras.layers.Layer):

    def __init__(self):
        super(WordAttention, self).__init__()

    def build(self, input_shape):  # TensorShape of input when run call(), inference from inputs
        self.mlp = keras.layers.Dense(input_shape[-1], activation=tf.nn.tanh)
        self.context = self.add_weight("context", shape=[1, input_shape[-1]], initializer='random_normal')

    def call(self, inputs, training=None):
        weight = self.mlp(inputs)
        weight = tf.matmul(self.context, weight, transpose_b=True)
        weight = tf.math.softmax(weight)
        outputs = tf.squeeze(tf.matmul(weight, inputs), axis=-2)
        if not training:
            print('WordAttention: weight: ', weight)
        return outputs

