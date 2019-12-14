import tensorflow as tf
import gin

AUTOTUNE = tf.data.experimental.AUTOTUNE


@gin.configurable
def dataset(batch_size, image_size=32, buffer_size=10000):
    train, test = tf.keras.datasets.mnist.load_data()
    train_images, test_images = train[0], test[0]

    def _process_image(image):
        image = tf.image.resize(image[:, :, None], (image_size, image_size))
        image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0

        return image

    train = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(buffer_size)
        .map(_process_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(buffer_size)
        .map(_process_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train, test
