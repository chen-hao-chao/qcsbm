import numpy as np
import io
import os
import time
import logging
import tensorflow as tf
import torch
from torchvision.utils import make_grid, save_image
from ml_collections.config_flags import config_flags
from absl import flags
from absl import app

import sampling
import datasets
import sde_lib
from models import ncsnpp, ncsnpp_mod
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import random

FLAGS = flags.FLAGS

def sampling_function(config, sampling_dir):
  config.model.num_scales = 1000
  # Create directories for experimental logs.
  sample_di_dir = os.path.join(sampling_dir, str(config.eval.division_idx))
  tf.io.gfile.makedirs(sample_di_dir)
  
  # Initialize models.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  checkpoint = torch.load(config.sampling.score_restore_path, map_location=config.device)
  score_model.load_state_dict(checkpoint['model'], strict=False)
  ema.load_state_dict(checkpoint['ema'])
  ema.copy_to(score_model.parameters())

  # Create data normalizer and its inverse.
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  config.model.num_scales = FLAGS.num_scales
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    eps = 1e-5
  else:
    raise NotImplementedError("SDE {} unknown.".format(config.training.sde))

  # Building sampling functions.
  sampling_shape = (config.eval.batch_size, config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, eps)
  # Conditionally generate the images for each class.
  division_size = config.eval.num_samples // config.eval.num_divisions
  num_sampling_rounds = division_size // config.eval.batch_size
  
  with torch.no_grad():
    for r in range(num_sampling_rounds):
      logging.info("Rounds: %d / %d." % (r+1, num_sampling_rounds))
      logging.info("Sample Index from %d to %d." % (config.eval.division_idx*division_size + r*config.eval.batch_size, config.eval.division_idx*division_size + (r+1)*config.eval.batch_size))
      now = time.time()
      # Generate the images using the sampling function.
      samples, nfe = sampling_fn(score_model)
      # Save the samples.
      nrow = int(np.sqrt(samples.shape[0]))
      image_grid = make_grid(samples, nrow, padding=2)
      samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
      samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
      with tf.io.gfile.GFile(os.path.join(sample_di_dir, "samples_{}.npz".format(r)), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, samples=samples, nfe=nfe)
        fout.write(io_buffer.getvalue())
      with tf.io.gfile.GFile(os.path.join(sample_di_dir, "samples_{}.png".format(r)), "wb") as fout:
        save_image(image_grid, fout)
      
      later = time.time()
      difference = int(later - now)
      logging.info("Time consumption: %d sec." % (difference))

config_flags.DEFINE_config_file("config", None, "Configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_integer("num_divisions", 10, "Number of GPUs avaliable for the sampling process.")
flags.DEFINE_integer("division_idx", 0, "Division index. (should be less than num_divisions)")
flags.DEFINE_integer("num_samples", 50000, "Total number of images in a dataset.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("bs", 2500, "Batch size.")
flags.DEFINE_integer("checkpoint_idx", 70, "The index of a checkpoint file.")
flags.DEFINE_boolean("ode", False, "Set as True to use an ODE solver.")
flags.DEFINE_integer("num_scales", 1000, "Number of function evaluation (NFE) used in a PC sampler.")
flags.DEFINE_bool("deterministic", False, "Set as true to make the sampling process deterministic.")

flags.mark_flags_as_required(["workdir", "config", "division_idx"])


def set_deterministic(config):
  config.seed = FLAGS.seed
  # Tensorflow
  tf.random.set_seed(config.seed)
  # OS
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
  # Pytorch
  torch.manual_seed(config.seed)
  torch.use_deterministic_algorithms(True)
  # Random
  random.seed(config.seed)
  # Numpy
  np.random.seed(config.seed)
  # Config
  config.training.deterministic = True

def main(argv):
  if FLAGS.deterministic:
    FLAGS.config.seed = 0
    set_deterministic(FLAGS.config)
  config = FLAGS.config
  workdir = os.path.join("results", FLAGS.workdir)
  dir_name = "eval_sampling" + ("_ode" if FLAGS.ode else "")
  sampling_dir = os.path.join(workdir, dir_name)
  tf.io.gfile.makedirs(sampling_dir)
  # Adjust the config file
  config.eval.division_idx = FLAGS.division_idx
  config.eval.num_divisions = FLAGS.num_divisions
  config.eval.num_samples = FLAGS.num_samples
  config.eval.batch_size = FLAGS.bs
  # Run the code
  config.sampling.score_restore_path = os.path.join(workdir, 'checkpoints/checkpoint_'+str(FLAGS.checkpoint_idx)+'.pth')
  sampling_function(config, sampling_dir)

if __name__ == "__main__":
  app.run(main)