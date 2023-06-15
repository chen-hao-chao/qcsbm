"""Entrance file"""
import run_lib, run_lib_eval
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or evaluation.")
flags.DEFINE_string("type", None, "Flag for switching different training / evaluation modes.")
flags.DEFINE_string("workdir", "checkerboard_qcsbm", "Directory for saving checkpoints and visualized results.")
flags.DEFINE_string("resultdir", "results", "Directory for saving workdir.")
flags.DEFINE_string("restore", "results/checkerboard_qcsbm/checkpoints/checkpoint_4000.pth", "The path to a checkpoint file.")
flags.mark_flags_as_required(["mode"])

def main(argv):
  if FLAGS.mode == "train":
    resultdir = FLAGS.resultdir
    tf.io.gfile.makedirs(resultdir)
    tf.io.gfile.makedirs(os.path.join(resultdir, FLAGS.workdir))
    run_lib.run(FLAGS.config, os.path.join(resultdir, FLAGS.workdir))
  elif FLAGS.mode == "eval":
    FLAGS.config.eval.type = FLAGS.type
    FLAGS.config.eval.restore_path = FLAGS.restore
    run_lib_eval.run(FLAGS.config)
  else:
    raise ValueError("Mode {} not recognized.".format(FLAGS.mode))

if __name__ == "__main__":
  app.run(main)