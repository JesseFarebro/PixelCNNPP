import tensorflow as tf

from layers.VerticalConv2DTranspose import VerticalConv2DTranspose
from layers.HorizontalConv2DTranspose import HorizontalConv2DTranspose


def test_vertical_masking():
    mock = tf.reshape(tf.range(9, dtype=tf.float32), (1, 3, 3, 1))

    v_conv = VerticalConv2DTranspose(
        1, 3, strides=2, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False
    )
    output = tf.squeeze(v_conv(mock))
    expected = tf.constant(
        [
            [0, 1, 1, 3, 2, 2],
            [0, 1, 1, 3, 2, 2],
            [3, 7, 4, 9, 5, 5],
            [3, 7, 4, 9, 5, 5],
            [6, 13, 7, 15, 8, 8],
            [6, 13, 7, 15, 8, 8],
        ],
        dtype=tf.float32,
    )
    assert output.shape == (6, 6)
    assert tf.math.reduce_all(tf.math.equal(output, expected))


def test_horizontal_masking():
    mock = tf.reshape(tf.range(9, dtype=tf.float32), (1, 3, 3, 1))

    h_conv = HorizontalConv2DTranspose(
        1, 3, strides=2, kernel_initializer=tf.keras.initializers.Ones()
    )
    output = tf.squeeze(h_conv(mock))
    expected = tf.constant(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
            [3, 3, 4, 4, 5, 5],
            [3, 3, 4, 4, 5, 5],
            [6, 6, 7, 7, 8, 8],
            [6, 6, 7, 7, 8, 8],
        ],
        dtype=tf.float32,
    )
    assert output.shape == (6, 6)
    assert tf.math.reduce_all(tf.math.equal(output, expected))
