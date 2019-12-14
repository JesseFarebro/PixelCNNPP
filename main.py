import atexit
import os
import platform
import re
import logging

import click
import gin

import tensorflow as tf
from train import train
from utils.helpers import gin_log_config, project_revision


@click.group(invoke_without_command=False)
@click.option(
    "--log-dir",
    default=None,
    show_default="logdir",
    type=click.Path(),
    help="Base directory to save logs and model checkpoints",
)
@click.option(
    "--debug", default=False, is_flag=True, help="Debug turns off graph compilation"
)
@click.option("--config", multiple=True, type=click.Path(exists=True))
@click.option("--binding", multiple=True, type=str, help="gin bindings")
@click.pass_context
def main(ctx, log_dir, debug, config, binding):
    """
    Parses the GIN config file given as the sole arguemnt to main
    and commences training.

    Creates a log directory if the option --log-dir was unspecified.
    """
    ctx.ensure_object(dict)
    logging.basicConfig(level=logging.INFO)

    # Parse config
    gin.parse_config_files_and_bindings(config, binding)

    slug = "#%s" % project_revision()
    if log_dir is None:
        # Generate new directory if not resuming training
        host = platform.node()

        version = 0
        prefix = os.path.join("logdir", "%s-%s" % (host, slug))
        if len(binding) > 0:
            prefix += "-%s" % "_".join(binding).strip(" ").replace("/", "\\")
        directory = "%s-%d" % (prefix, version)
        while tf.io.gfile.isdir(directory):
            version += 1
            directory = "%s-%d" % (prefix, version)
        log_dir = directory

    if not tf.io.gfile.isdir(log_dir):
        tf.io.gfile.makedirs(log_dir)
    atexit.register(gin_log_config, log_dir, slug)

    if debug:
        logging.info("Debug enabled, disabling graph compilation")

    ctx.obj["log_dir"] = log_dir
    ctx.obj["debug"] = debug

    return ctx


@main.command()
@click.pass_context
def multigpu(ctx):
    return ctx, tf.distribute.MirroredStrategy()


@main.command()
@click.option("--device", default=None, type=str)
@click.pass_context
def single(ctx, device):
    if device is None:
        device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"

    return ctx, tf.distribute.OneDeviceStrategy(device)


@main.resultcallback(replace=True)
def callback(args, **kwargs):
    ctx, strategy = args
    log_dir = ctx.obj["log_dir"]
    logging.info("Using log directory: %s" % log_dir)

    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        train(strategy, **ctx.obj)
    logging.info("Finished training")


if __name__ == "__main__":
    main(obj={})
