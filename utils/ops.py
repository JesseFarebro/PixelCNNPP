import tensorflow as tf

from math import factorial


def concat_elu(inputs):
    return tf.nn.elu(tf.concat([inputs, -inputs], axis=-1))

def binomial_2(n):
    n = max(n, 2)
    return factorial(n) // (2 * factorial(n - 2))

def down_shift(inputs):
    """
    Pop the last row and shift in zeros from the top

    inputs: (B, H, W, N) where
            B: Batch size
            H: Height
            W: Width
            N: Number of channels
    outputs: (B, H, W, N) where [:, 0, :, :] is 0s
             and [:, i, :, :] = [:, i - 1, :, :] for i = H ... 1
    """
    cropped = inputs[:, :-1, :, :]

    paddings = [
        [0, 0],  # Before dim 0  # After dim 0
        [1, 0],  # Before dim 1  # After dim 1
        [0, 0],  # Before dim 2  # After dim 2
        [0, 0],  # Before dim 3  # After dim 3
    ]
    padded = tf.pad(cropped, paddings)

    return padded


def right_shift(inputs):
    """
    Pop the last column and shift in zeros from the left

    inputs: (B, H, W, N) where
            B: Batch size
            H: Height
            W: Width
            N: Number of channels
    outputs: (B, H, W, N) where [:, :, 0, :] is 0s
             and [:, :, i, :] = [:, :, i - 1, :] for i = W ... 1
    """
    cropped = inputs[:, :, :-1, :]

    paddings = [
        [0, 0],  # Before dim 0  # After dim 0
        [0, 0],  # Before dim 1  # After dim 1
        [1, 0],  # Before dim 2  # After dim 2
        [0, 0],  # Before dim 3  # After dim 3
    ]
    padded = tf.pad(cropped, paddings)

    return padded
