from utils import get_distribution
import tensorflow as tf


class Layer:
    def __init__(self, layer_settings, size=(None, None)):
        self.layer_settings = layer_settings
        self._size = size
        self.in_size = size[0]
        self.out_size = size[1]
        if self.layer_settings:
            self._activation_function = get_activation_function(name=self.layer_settings.activation)
        else:
            self._activation_function = None
        self._input = None
        self._output_before_activation = None
        self._output = None
        self.weights = None
        self.biases = None
        self._net = None
        self._name = None

    def setup(self, net, name):
        self._net = net
        self._name = name

    def get_layer_output(self):
        return self._output

    def get_layer_output_before_activation(self):
        return self._output_before_activation

    def get_input(self):
        return self._input

    def get_name(self):
        return self._name


class InputLayer(Layer):
    def build_layer(self):
        self._output_before_activation = tf.placeholder(tf.float64, shape=self._size, name='Input')
        if self.layer_settings.normalize:
            self._output = tf.nn.l2_normalize(self._output_before_activation)
        else:
            self._output = self._output_before_activation


class DenseLayer(Layer):
    def build_layer(self, layer_input):
        self._input = layer_input

        self.weights = tf.Variable(
            get_distribution(self.layer_settings.weights, self._size),
            name=self._name + '-weights',
        )
        self.biases = tf.Variable(
            get_distribution(self.layer_settings.biases, self._size[1]),
            name=self._name + '-bias',
        )
        self._output_before_activation = tf.matmul(self._input, self.weights) + self.biases
        if self._activation_function:
            self._output = self._activation_function(self._output_before_activation, name=self._name + '-out')
        else:
            self._output = self._output_before_activation


def get_activation_function(name):
    if name == 'relu':
        return tf.nn.relu
    elif name == 'softmax':
        return tf.nn.softmax
    elif name == 'sigmoid':
        return lambda x, name: tf.round(tf.nn.sigmoid(x, name))
    elif name == 'tanh':
        return tf.nn.tanh
    else:
        return None
