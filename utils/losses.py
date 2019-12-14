import tensorflow as tf

from utils.ops import binomial_2


def log_probs_from_logits(x):
    # Uses the exp-normalization trick for numerical stability
    b = tf.math.reduce_max(x, axis=-1, keepdims=True)
    return x - b - tf.math.reduce_logsumexp(x - b, axis=-1, keepdims=True)


# #Outputs = #mixtures + (#mixtures * channels * 2) + (#mixtures * binomial(channels, 2))
def logistic_mixture_loss(samples, mixture, num_mixtures=10):
    _, width, height, num_channels = samples.shape.as_list()

    num_coeffs = binomial_2(num_channels)
    num_logistic_params = num_channels * num_mixtures * 2

    # Step 1: Extracting parameters
    #
    # Weightings for each mixture
    pi = mixture[:, :, :, :num_mixtures]

    # The three parameters for each mixture are:
    #   [means, log scales, scalings]
    #
    #   Mean and log scale correspond to the mean and log scale for
    #   the logistic distribution.
    #
    #   Scalings are the linear scalings seen in Eq. 3 of PixelCNN++
    #   with alpha, beta, and gamma.
    #
    logistic_params = tf.reshape(
        mixture[:, :, :, num_mixtures : num_mixtures + num_logistic_params],
        [-1, width, height, num_channels, num_logistic_params // num_channels],
    )
    means, log_scales = tf.split(logistic_params, 2, axis=-1)

    coeffs = tf.reshape(
        mixture[:, :, :, (num_mixtures + num_logistic_params) :],
        [-1, width, height, num_coeffs, num_mixtures],
    )
    coeffs = tf.nn.tanh(coeffs)

    # Step 2: Scaling
    #
    # Following the PixelCNN paper we will now scale the pixel predictions
    # for each subpixel allowing linear dependencies on previous channels.
    # This is precisely Eq. 3 in the paper.
    #
    #
    channel_means = tf.split(means, num_channels, axis=3)

    for i in range(num_channels):
        coeff_start = sum(range(i))
        for j in range(i - 1, -1, -1):
            channel_means[i] += (
                samples[:, :, :, j, tf.newaxis, tf.newaxis]
                * coeffs[:, :, :, tf.newaxis, coeff_start + j, :]
            )

    means = tf.concat(channel_means, axis=-2)

    # Step 3: Discretization
    #
    # i.e. CDF+ = F(x + h)
    #      CDF- = F(x - h)
    #   P(y - h < x < y + h) = CDF+ - CDF-
    #
    h = 1.0 / 255

    # The scales from the logisitc distribution must be positive.
    # To accomplish this we output log scales and exponentiate.
    # For stability we lower bound the log scales.
    # We also need 1/scales so we'll just use negate the scales
    # before exponentiation.
    inv_scales = tf.math.exp(-tf.math.maximum(log_scales, -7.0))

    # Reshape and center the samples
    samples = tf.tile(samples[:, :, :, :, tf.newaxis], [1, 1, 1, 1, num_mixtures])
    centered = samples - means

    plus = inv_scales * (centered + h)
    mid = inv_scales * centered
    minus = inv_scales * (centered - h)

    cdf_plus = tf.math.sigmoid(plus)
    cdf_minus = tf.math.sigmoid(minus)

    discretized_log_probs = cdf_plus - cdf_minus

    # Edge case of 0
    # We need to discretize the edge case of 0:
    #   F(x + \Delta x; mean, scale) - F(-inf; mean, scale)
    #
    #   Note the limit of F(-inf, mean, scale) -> 0
    #
    #   So we need the log probability of F(x + \Delta x; mean, scale)
    #   We can use some identities of the logistic sigmoid namely:
    #       log(sigmoid(x)) = -softplus(-x)
    #       softplus(x) - softplus(-x) = x
    #
    left_cdf_edge = plus - tf.math.softplus(plus)

    # Edge case of 255
    # We need to discretize the edge case of 255:
    #   F(inf; mean, scale) - F(x - \Delta x; mean, scale)
    #
    #   Note the limit of F(inf, mean, scale) -> 1
    #
    #   So we need the log probability of 1 - F(x - \Delta x; mean, scale)
    #   We can use some identities of the logistic sigmoid namely:
    #       1 - sigmoid(x) = sigmoid(-x)
    #       log(sigmoid(x)) = -softplus(-x)
    #
    right_cdf_edge = -tf.math.softplus(minus)

    # If we fall below a threshold when computing
    #    F(x + \Delta x) - F(x - \Delta x)
    # we will use an approximation to the integral using
    # the midpoint method.
    # i.e., f(x; mean, scale) * \delta x
    # The log of the logistic PDF reduces to:
    #   x - log(scale) - 2 * log(softplus(x))
    #
    pdf_mid = mid - log_scales - 2.0 * tf.math.softplus(mid)

    # Step 4: Construction the loss
    #
    # We clamp the values less than -0.999 (0 pixel edge case)
    # to our left cdf edge. We do the same thing for 0.999 (255 pixel case)
    # with our right cdf edge.
    #
    # If we don't fall into either the left or right edge case
    # then we use the probability values from F(x + \Delta x) - F(x - \Delta x)
    #
    # If the probability values are below a threshold then we use "an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value" as per openai/pixel-cnn.
    #
    log_probs = tf.where(
        samples < -0.999,
        left_cdf_edge,
        tf.where(
            samples > 0.999,
            right_cdf_edge,
            tf.where(
                discretized_log_probs > 1e-5,
                tf.math.log(tf.maximum(discretized_log_probs, 1e-12)),
                pdf_mid - tf.math.log(127.5),
            ),
        ),
    )

    # Weight by the mixture factors \pi
    log_probs = tf.math.reduce_sum(log_probs, axis=3) + log_probs_from_logits(pi)

    # Log likelihood reduced over all axes b.c. autoregressive factorization
    loss = -tf.math.reduce_sum(tf.math.reduce_logsumexp(log_probs, axis=-1))

    return loss
