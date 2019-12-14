import tensorflow as tf

from layers.ResidualBlock import ResidualBlock

from layers.VerticalConv2D import VerticalConv2D
from layers.VerticalConv2DTranspose import VerticalConv2DTranspose

from layers.HorizontalConv2D import HorizontalConv2D
from layers.HorizontalConv2DTranspose import HorizontalConv2DTranspose

from layers.WeightNormalization import WeightNormalization

from utils.ops import concat_elu


class PixelCNNBlock(tf.keras.Model):
    """
    PixelCNNBlock consists of multiple PixelCNNLayers.

    This can be seen by the boxes outlining blocks in Figure 2 in the paper:
    "PixelCNN++: Improving the PixelCNN with
        Discretized Logistic Mixture Likelihood and Other Modifications"
    by Salimans et al.
    https://arxiv.org/pdf/1701.05517.pdf

    PixelCNNBlock can optionally return the output of each
    PixelCNNLayer to be used for U-net like skip connections.
    """

    def __init__(
        self,
        num_layers,
        filters,
        kernel_size,
        resample=None,
        resample_strides=(2, 2),
        dropout_rate=0.5,
        activation=concat_elu,
        **kwargs
    ):
        super(PixelCNNBlock, self).__init__(**kwargs)
        self.resample = resample
        self.num_layers = num_layers

        self.vertical_stream = [
            ResidualBlock(
                filters,
                kernel_size,
                Conv2D=VerticalConv2D,
                dropout_rate=dropout_rate,
                activation=activation,
            )
            for _ in range(num_layers)
        ]

        self.horizontal_stream = [
            ResidualBlock(
                filters,
                kernel_size,
                Conv2D=HorizontalConv2D,
                dropout_rate=dropout_rate,
                activation=activation,
            )
            for _ in range(num_layers)
        ]

        if isinstance(resample, str) and resample.lower() == "down":
            self.vertical_resample = WeightNormalization(
                VerticalConv2D(filters, kernel_size, strides=resample_strides)
            )
            self.horizontal_resample = WeightNormalization(
                HorizontalConv2D(filters, kernel_size, strides=resample_strides)
            )
        elif isinstance(resample, str) and resample.lower() == "up":
            self.vertical_resample = WeightNormalization(
                VerticalConv2DTranspose(filters, kernel_size, strides=resample_strides)
            )
            self.horizontal_resample = WeightNormalization(
                HorizontalConv2DTranspose(
                    filters, kernel_size, strides=resample_strides
                )
            )
        else:
            self.resample = None
            self.vertical_resample = lambda x: x
            self.horizontal_resample = lambda x: x

    def call(self, vertical, horizontal, skip=None, all_outputs=False, training=False):
        vertical_outputs, horizontal_outputs = [], []

        for layer in range(self.num_layers):
            vertical = self.vertical_stream[layer](
                vertical,
                skip=skip[0].pop() if skip is not None else None,
                training=training,
            )
            horizontal = self.horizontal_stream[layer](
                horizontal,
                skip=tf.concat((vertical, skip[1].pop()), axis=-1)
                if skip is not None
                else vertical,
                training=training,
            )

            if all_outputs:
                vertical_outputs += [vertical]
                horizontal_outputs += [horizontal]

        vertical = self.vertical_resample(vertical)
        horizontal = self.horizontal_resample(horizontal)

        if self.resample is not None and all_outputs:
            vertical_outputs.append(vertical)
            horizontal_outputs.append(horizontal)

        if all_outputs:
            return vertical_outputs, horizontal_outputs

        return vertical, horizontal
