import tensorflow as tf
import gin
import logging

from tqdm import trange, tqdm

from models.PixelCNNPP import PixelCNNPP
from utils.losses import logistic_mixture_loss


@gin.configurable
def train(
    strategy,
    log_dir,
    dataset_fn,
    model_cls=PixelCNNPP,
    optimizer_cls=tf.keras.optimizers.Adam,
    learning_rate=0.0002,
    learning_rate_decay=0.999995,
    batch_size=64,
    max_epoch=5000,
    chkpt_to_keep=5,
    images_to_log=16,
    log_images_every=50,
    debug=False,
    **kwargs
):
    logging.info("Running with %d replicas" % strategy.num_replicas_in_sync)
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    train_dataset, eval_dataset = dataset_fn(global_batch_size)

    train_len = tf.data.experimental.cardinality(train_dataset)
    eval_len = tf.data.experimental.cardinality(eval_dataset)

    train_iterator = strategy.experimental_distribute_dataset(train_dataset)
    eval_iterator = strategy.experimental_distribute_dataset(eval_dataset)

    structure = tf.data.experimental.get_structure(train_iterator)
    _, width, height, channels = structure.shape.as_list()
    inputs_shape = tf.TensorShape([None, width, height, channels])

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, max_epoch, learning_rate_decay
    )

    with strategy.scope():
        model = model_cls(inputs_shape)
        model.build(inputs_shape)

        optimizer = optimizer_cls(learning_rate_schedule)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, log_dir, chkpt_to_keep, 1)
    restore_status = checkpoint.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        logging.info("Resuming from %s" % manager.latest_checkpoint)
        restore_status.assert_existing_objects_matched()

    with strategy.scope():

        @tf.function
        def train_step(batch):
            def step_fn(inputs):
                with tf.GradientTape() as tape:
                    mixture = model(inputs, training=True)
                    loss = logistic_mixture_loss(
                        inputs, mixture, num_mixtures=model.num_mixtures
                    )

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                return loss

            per_replica_loss = strategy.experimental_run_v2(step_fn, (batch,))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None
            )

        @tf.function
        def eval_step(batch):
            def step_fn(inputs):
                mixture = model(inputs, training=False)
                loss = logistic_mixture_loss(
                    inputs, mixture, num_mixtures=model.num_mixtures
                )

                return loss

            per_replica_loss = strategy.experimental_run_v2(step_fn, (batch,))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None
            )

    bpd = lambda loss: loss / (
        global_batch_size * tf.math.log(2.0) * width * height * channels
    )
    train_loss = tf.keras.metrics.Mean("train_loss")
    train_bpd = tf.keras.metrics.Mean("train_bpd")

    eval_loss = tf.keras.metrics.Mean("eval_loss")
    eval_bpd = tf.keras.metrics.Mean("eval_bpd")
    for epoch in trange(1, max_epoch + 1, initial=1):
        train_loss.reset_states()
        train_bpd.reset_states()

        for batch in tqdm(
            train_iterator,
            total=train_len.numpy() if train_len > 0 else None,
            desc="train",
            unit="images",
            unit_scale=global_batch_size,
        ):
            aggregate_loss = train_step(batch)

            train_loss.update_state(aggregate_loss)
            train_bpd.update_state(bpd(aggregate_loss))

        eval_loss.reset_states()
        eval_bpd.reset_states()
        for batch in tqdm(
            eval_iterator,
            total=eval_len.numpy() if eval_len > 0 else None,
            desc="eval",
            unit="images",
            unit_scale=global_batch_size,
        ):
            aggregate_loss = eval_step(batch)
            eval_loss.update_state(aggregate_loss)
            eval_bpd.update_state(bpd(aggregate_loss))

        tf.summary.scalar(
            "train/NegativeLogLikelihood", train_loss.result(), step=epoch
        )
        tf.summary.scalar("train/BitsPerDimension", train_bpd.result(), step=epoch)
        tf.summary.scalar("eval/NegaitveLogLikelihood", eval_loss.result(), step=epoch)
        tf.summary.scalar("eval/BitsPerDimension", eval_bpd.result(), step=epoch)

        if epoch % log_images_every == 0:
            samples = model.sample(images_to_log)
            samples = tf.cast((samples + 1.0) * 127.5, tf.uint8)
            tf.summary.image("samples", samples, step=epoch, max_outputs=images_to_log)

        manager.save(epoch)
