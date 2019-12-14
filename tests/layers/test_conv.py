import tensorflow as tf

from layers.VerticalConv2D import VerticalConv2D
from layers.HorizontalConv2D import HorizontalConv2D


def test_vertical_masking():
    mock = tf.reshape(tf.range(25, dtype=tf.float32), (1, 5, 5, 1))

    v_conv = VerticalConv2D(
        1, 3, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False
    )
    output = tf.squeeze(v_conv(mock))
    expected = tf.constant(
        [
            [1, 3, 6, 9, 7],
            [12, 21, 27, 33, 24],
            [32, 51, 57, 63, 44],
            [52, 81, 87, 93, 64],
            [72, 111, 117, 123, 84],
        ],
        dtype=tf.float32,
    )
    assert output.shape == (5, 5)
    assert tf.math.reduce_all(tf.math.equal(output, expected))

    v_conv = VerticalConv2D(
        1,
        3,
        kernel_initializer=tf.keras.initializers.Ones(),
        down_shift=True,
        use_bias=False,
    )
    output = tf.squeeze(v_conv(mock))
    expected = tf.constant(
        [
            [0, 0, 0, 0, 0],
            [1, 3, 6, 9, 7],
            [12, 21, 27, 33, 24],
            [32, 51, 57, 63, 44],
            [52, 81, 87, 93, 64],
        ],
        dtype=tf.float32,
    )
    assert output.shape == (5, 5)
    assert tf.math.reduce_all(tf.math.equal(output, expected))


def test_horizontal_masking():
    mock = tf.reshape(tf.range(25, dtype=tf.float32), (1, 5, 5, 1))

    h_conv = HorizontalConv2D(
        1, 3, kernel_initializer=tf.keras.initializers.Ones(), use_bias=False
    )
    output = tf.squeeze(h_conv(mock))
    expected = tf.constant(
        [
            [0, 1, 3, 5, 7],
            [5, 12, 16, 20, 24],
            [15, 32, 36, 40, 44],
            [25, 52, 56, 60, 64],
            [35, 72, 76, 80, 84],
        ],
        dtype=tf.float32,
    )
    assert output.shape == (5, 5)
    assert tf.math.reduce_all(tf.math.equal(output, expected))
