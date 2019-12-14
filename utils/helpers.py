import datetime
import os
import subprocess
import logging

import gin

import tensorflow as tf


def project_revision():
    """
    Returns git short sha256 or current datetime if we can't execute git
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except:
        return datetime.datetime.now().strftime("%y%m%d:%H")


def gin_log_config(log_dir, slug):
    """
    Attempt to write the opperative config to our log dir and Tensorboard...
    Also prints the config to tf.logging.INFO
    """
    try:
        if log_dir is not None and tf.io.gfile.isdir(log_dir):
            path = os.path.join(log_dir, "%s.gin" % slug)
            with tf.io.gfile.GFile(path, "w") as fp:
                fp.write(gin.operative_config_str())
    except:
        logging.info(gin.operative_config_str())
        logging.warn("Failed to write log gin config")
