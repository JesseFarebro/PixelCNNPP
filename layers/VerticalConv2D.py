import tensorflow as tf

from utils.ops import down_shift as down_shift_op


class VerticalConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, down_shift=False, **kwargs):
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size // 2 + 1, kernel_size)
        assert (
            isinstance(kernel_size, tuple) and len(kernel_size) == 2
        ), "Kernel size must be  a tuple of two integers"
        super(VerticalConv2D, self).__init__(filters, kernel_size, **kwargs)
        self.pad = tf.keras.layers.ZeroPadding2D(
            (
                (kernel_size[0] - 1, 0),  # Top, Bottom
                (kernel_size[1] // 2, kernel_size[1] // 2),  # Left, Right
            )
        )
        self.crop = tf.keras.layers.Lambda(
            lambda x: down_shift_op(x) if down_shift else x
        )

    def call(self, inputs):
        inputs = self.pad(inputs)
        output = super(VerticalConv2D, self).call(inputs)

        return self.crop(output)
