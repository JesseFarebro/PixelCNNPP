import tensorflow as tf
import gin

from layers.PixelCNNBlock import PixelCNNBlock
from layers.VerticalConv2D import VerticalConv2D
from layers.HorizontalConv2D import HorizontalConv2D

from layers.WeightNormalization import WeightNormalization

from utils.ops import concat_elu, binomial_2


@gin.configurable
class PixelCNNPP(tf.keras.Model):
    def __init__(
        self,
        input_shape,
        residual_blocks=3,
        residual_layers_per_block=3,
        filters=80,
        kernel_size=3,
        dropout_rate=0.5,
        num_mixtures=10,
        activation=concat_elu,
        **kwargs
    ):
        super(PixelCNNPP, self).__init__(**kwargs)
        self.num_mixtures = num_mixtures
        self.activation = activation

        self.inputs_shape = input_shape

        self.v2v_init = WeightNormalization(
            VerticalConv2D(filters, kernel_size, down_shift=True)
        )
        self.v2h_init = WeightNormalization(
            VerticalConv2D(filters, (kernel_size // 2, kernel_size), down_shift=True)
        )
        self.h2h_init = WeightNormalization(
            HorizontalConv2D(
                filters, ((kernel_size + 1) // 2, kernel_size // 2), right_shift=True
            )
        )

        # Only resample on the first n - 1 blocks
        self.down_blocks = [
            PixelCNNBlock(
                residual_layers_per_block,
                filters,
                kernel_size,
                resample="down" if i < residual_blocks - i else None,
                activation=activation,
                dropout_rate=dropout_rate,
            )
            for i in range(residual_blocks)
        ]

        # Only resample on the first n - 1 blocks
        self.up_blocks = [
            PixelCNNBlock(
                residual_layers_per_block if i == 0 else residual_layers_per_block + 1,
                filters,
                kernel_size,
                resample="up" if i < residual_blocks - 1 else None,
                activation=activation,
                dropout_rate=dropout_rate,
            )
            for i in range(residual_blocks)
        ]

        # #mixtures + (#mixtures * channels * 2) + (#mixtures * binomial(channels, 2))
        num_channels = input_shape[-1]
        num_params = (
            num_mixtures
            + (num_mixtures * num_channels * 2)
            + (num_mixtures * binomial_2(num_channels))
        )
        self.output_conv = WeightNormalization(tf.keras.layers.Conv2D(num_params, 1))

    def call(self, inputs, training=False):
        # Init convs
        vertical_stream = [self.v2v_init(inputs)]
        horizontal_stream = [self.v2h_init(inputs) + self.h2h_init(inputs)]

        # Down pass
        for down_block in self.down_blocks:
            vertical, horizontal = down_block(
                vertical_stream[-1],
                horizontal_stream[-1],
                all_outputs=True,
                training=training,
            )

            # Add stream values for skips later
            vertical_stream.extend(vertical)
            horizontal_stream.extend(horizontal)

        vertical, horizontal = vertical_stream.pop(), horizontal_stream.pop()
        # Up pass
        for up_block in self.up_blocks:
            vertical, horizontal = up_block(
                vertical,
                horizontal,
                skip=(vertical_stream, horizontal_stream),
                training=training,
            )

        assert (
            len(vertical_stream) == 0 and len(horizontal_stream) == 0
        ), "Vertical or horizontal stream is not empty"

        return self.output_conv(self.activation(horizontal))

    @tf.function
    def _sample_from_mixture(self, inputs):
        _, width, height, num_channels = inputs.shape
        num_coeffs = binomial_2(num_channels)
        num_logistic_params = num_channels * self.num_mixtures * 2

        mixture = self.call(inputs, training=False)
        pi = mixture[:, :, :, : self.num_mixtures]

        # Gumbel-max trick to sample from discrete distribution
        gumbel = -tf.math.log(
            -tf.math.log(tf.random.uniform(pi.shape, minval=1e-5, maxval=1.0 - 1e-5))
        )

        selected = tf.one_hot(
            tf.argmax(pi + gumbel, axis=-1), depth=self.num_mixtures, dtype=tf.float32
        )
        selected = tf.expand_dims(selected, axis=-2)

        logistic_params = tf.reshape(
            mixture[
                :, :, :, self.num_mixtures : self.num_mixtures + num_logistic_params
            ],
            [-1, width, height, num_channels, num_logistic_params // num_channels],
        )
        means, log_scales = tf.split(logistic_params, 2, axis=-1)

        means = tf.reduce_sum(means * selected, axis=-1)
        log_scales = tf.reduce_sum(log_scales * selected, axis=-1)

        # Inverse sampling
        scales = tf.math.exp(tf.math.maximum(log_scales, -7.0))
        p = tf.random.uniform(means.shape, minval=1e-5, maxval=1.0 - 1e-5)

        # Logistic quantile with log division rule for numerical stability
        x = means + scales * (tf.math.log(p) - tf.math.log(1 - p))

        coeffs = tf.reshape(
            tf.nn.tanh(mixture[:, :, :, (self.num_mixtures + num_logistic_params) :]),
            [-1, width, height, num_coeffs, self.num_mixtures],
        )
        coeffs = tf.reduce_sum(coeffs * selected, axis=-1, keepdims=True)

        channels = tf.split(x, num_channels, axis=-1)
        for i in range(num_channels):
            coeff_start = sum(range(i))
            for j in range(i - 1, -1, -1):
                channels[i] += channels[j] * coeffs[:, :, :, coeff_start + j, :]
            channels[i] = tf.clip_by_value(channels[i], -1.0, 1.0)

        return tf.concat(channels, axis=-1)

    def sample(self, batch_size):
        _, width, height, channels = self.inputs_shape
        images = tf.zeros((batch_size, width, height, channels)).numpy()

        for i in range(width):
            for j in range(height):
                sample = self._sample_from_mixture(images)
                images[:, i, j, :] = sample[:, i, j, :]

        return images
