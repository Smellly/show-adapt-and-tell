import tensorflow as tf

import sys
sys.path.append('/home/smelly/miniconda2/envs/tensorflow/lib/python2.7/site-packages/tensorflow/python/ops')
import rnn_cell_impl

# highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in xrange(layer_size):
        # output = f(tf.nn.rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))
        with tf.variable_scope('output_lin_%d' % idx):
            output = f(
                    rnn_cell_impl._linear(
                        output, size, 0))
            # for tf.1.2.1 above
            # tf.nn.rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
        with tf.variable_scope('transform_lin_%d' % idx):
            transform_gate = tf.sigmoid(
                rnn_cell_impl._linear(
                    input_, size, 0) + bias)
        carry_gate = 1. - transform_gate
        output = transform_gate * output + carry_gate * input_
    return output

