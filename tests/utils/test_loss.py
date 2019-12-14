import tensorflow as tf
from utils.losses import logistic_mixture_loss
from utils.ops import binomial_2


def loss(channels, mixtures=10):
    params = mixtures + (mixtures * channels * 2) + (mixtures * binomial_2(channels))

    tf.random.set_seed(0)

    inputs = tf.random.uniform((64, 32, 32, channels), minval=-1, maxval=1)
    outputs = tf.random.uniform((64, 32, 32, params), minval=-1, maxval=1)

    return logistic_mixture_loss(inputs, outputs, num_mixtures=mixtures)


def test_logistic_mixture_loss_4c():
    output = loss(4)
    expected = tf.constant(1677945.0, dtype=tf.float32)
    assert tf.reduce_all(tf.math.equal(output, expected))


def test_logistic_mixture_loss_3c():
    output = loss(3)
    expected = tf.constant(1256211.5, dtype=tf.float32)
    assert tf.reduce_all(tf.math.equal(output, expected))


def test_logistic_mixture_loss_1c():
    output = loss(1)
    expected = tf.constant(417330.97, dtype=tf.float32)
    assert tf.reduce_all(tf.math.equal(output, expected))
