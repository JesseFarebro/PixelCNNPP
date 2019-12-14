import tensorflow as tf

from layers.HorizontalConv2D import HorizontalConv2D
from layers.VerticalConv2D import VerticalConv2D
from layers.WeightNormalization import WeightNormalization

from utils.ops import concat_elu


class ResidualBlock(tf.keras.Model):
    """
    A PixelCNNLayer is a layer which takes as input both the
    vertical and horizontal stream of pixels and performs
    masked convolutions using padding and cropping in order
    to maintain the conditional distribution of PixelCNN.

    This layer performs the computation outlined in Figure 2 of
    "Conditional Image Generation with PixelCNN Decoders" by van den Oord et al
    https://arxiv.org/pdf/1606.05328.pdf
    """

    def __init__(
        self,
        filters,
        kernel_size,
        Conv2D=tf.keras.layers.Conv2D,
        dropout_rate=0.5,
        activation=concat_elu,
        **kwargs
    ):
        super(ResidualBlock, self).__init__(**kwargs)

        self.activation = activation
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Input masked convolution
        self.conv_input = WeightNormalization(Conv2D(filters, kernel_size))

        # Output masked convolution to perform gating
        self.conv_output = WeightNormalization(Conv2D(filters * 2, kernel_size))

        # Skip 1x1 conv
        self.skip_resample = WeightNormalization(tf.keras.layers.Conv2D(filters, 1))

    def call(self, inputs, skip=None, training=False):
        stream = self.conv_input(self.activation(inputs))

        if skip is not None:
            stream += self.skip_resample(self.activation(skip))

        # Nonlinearity + Dropout
        stream = self.dropout(self.activation(stream), training=training)

        # Output convolution
        stream = self.conv_output(stream)

        # Split feature maps
        left, right = tf.split(stream, 2, axis=-1)

        # Linear gating from Dauphin et al 2017
        stream = left * tf.sigmoid(right)

        # Skip connection
        return stream + inputs
