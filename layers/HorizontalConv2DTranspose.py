import tensorflow as tf


class HorizontalConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self, filters, kernel_size, **kwargs):
        super(HorizontalConv2DTranspose, self).__init__(
            filters, kernel_size // 2 + 1, output_padding=1, **kwargs
        )
        self.crop = tf.keras.layers.Cropping2D(
            ((0, kernel_size // 2), (0, kernel_size // 2))
        )

    def call(self, inputs):
        output = super(HorizontalConv2DTranspose, self).call(inputs)

        return self.crop(output)
