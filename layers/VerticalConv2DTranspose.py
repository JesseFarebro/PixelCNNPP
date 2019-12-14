import tensorflow as tf


class VerticalConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self, filters, kernel_size, **kwargs):
        super(VerticalConv2DTranspose, self).__init__(
            filters, (kernel_size // 2 + 1, kernel_size), output_padding=1, **kwargs
        )
        self.crop = tf.keras.layers.Cropping2D(
            (
                (0, kernel_size // 2),  # (Top, Bottom)
                (kernel_size // 2, kernel_size // 2),  # (Left, Right)
            )
        )

    def call(self, inputs):
        output = super(VerticalConv2DTranspose, self).call(inputs)
        return self.crop(output)
