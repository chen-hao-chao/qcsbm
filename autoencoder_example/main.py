"""Entrance file"""
import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os
import torch
import random

# set_deterministic
def set_deterministic(config):
  # Pytorch
  torch.manual_seed(config.seed)
  # Numpy
  np.random.seed(config.seed)
  # Random
  random.seed(config.seed)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", "ae_example", "Directory for saving visualized results.")
flags.DEFINE_string("resultdir", "results", "Directory for saving workdir.")
flags.DEFINE_bool("deterministic", False, "Set true to make the result deterministic.")
flags.DEFINE_integer("seed", 0, "Set the seed.")

def main(argv):
  if FLAGS.deterministic:
    FLAGS.config.seed = FLAGS.seed
    set_deterministic(FLAGS.config)

  resultdir = FLAGS.resultdir
  tf.io.gfile.makedirs(resultdir)
  tf.io.gfile.makedirs(os.path.join(resultdir, FLAGS.workdir))
  run_lib.run(FLAGS.config, os.path.join(resultdir, FLAGS.workdir))

if __name__ == "__main__":
  app.run(main)