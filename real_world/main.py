import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_integer("slices", 1, "Number of random vectors used in Hutchinson's trace estimator (K).")
flags.DEFINE_integer("bs", 128, "Batch size (b).")
flags.DEFINE_float("balancing_fac", 0.0001, "Balancing factor (lambda).")
flags.DEFINE_string("restore", None, "The path to a checkpoint file.")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
  FLAGS.workdir = os.path.join("results", FLAGS.workdir)
  tf.io.gfile.makedirs(FLAGS.workdir)
  FLAGS.config.loss.slices = FLAGS.slices
  FLAGS.config.loss.balancing_fac = FLAGS.balancing_fac
  FLAGS.config.training.batch_size = FLAGS.bs
  FLAGS.config.eval.batch_size = FLAGS.bs
  FLAGS.config.training.restore_path = FLAGS.restore
  gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
  handler = logging.StreamHandler(gfile_stream)
  formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
  handler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.addHandler(handler)
  logger.setLevel('INFO')
  run_lib.train(FLAGS.config, FLAGS.workdir)

if __name__ == "__main__":
  app.run(main)
