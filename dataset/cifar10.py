import tensorflow as tf
import gin

AUTOTUNE = tf.data.experimental.AUTOTUNE


@gin.configurable
def dataset(batch_size, shuffle_buffer_size=10000):
    train, test = tf.keras.datasets.cifar10.load_data()
    train_images, test_images = train[0], test[0]

    def _process_image(image):
        return tf.cast(image, tf.float32) * (2.0 / 255) - 1.0

    train = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .map(_process_image, num_parallel_calls=AUTOTUNE)
        .shuffle(shuffle_buffer_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .map(_process_image, num_parallel_calls=AUTOTUNE)
        .shuffle(shuffle_buffer_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train, test
