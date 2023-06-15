import logging
import numpy as np
import os
import io
import time
import tensorflow as tf
import torch
from ml_collections.config_flags import config_flags
from absl import flags
from absl import app

import losses
import datasets
import sde_lib
from models import ncsnpp, ncsnpp_mod
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, get_sym_function


FLAGS = flags.FLAGS

def calculate_asym(config, workdir):
  # Create folder
  sym_dir = os.path.join(workdir, "asym")
  tf.io.gfile.makedirs(sym_dir)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  state = restore_checkpoint(config.sampling.score_restore_path, state, config.device)
  score_model.eval()

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create data loaders for Asym / NAsym evaluation. Only evaluate on uniformly dequantized data
  _, ds_asym, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization, evaluation=True)

  slices = config.loss.slices
  energy = config.model.energy
  continuous = config.training.continuous
  asym_fn = get_sym_function(sde=sde, continuous=continuous, energy=energy, slices=slices)

  for repeat in range(config.eval.total_repeat):
    asym_nl = []
    nasym_nl = []
    noise_levels = np.arange(0, 1, 1/config.eval.noise_levels)
    for nl in range(len(noise_levels)):
      asym_list = []
      nasym_list = []
      asym_iter = iter(ds_asym)
      now = time.time()
      for batch_id in range(len(ds_asym)):
        batch = next(asym_iter)
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        asym_, nasym_ = asym_fn(score_model, eval_batch, noise_levels[nl])
        asym_ = asym_.detach().cpu().numpy().reshape(-1)
        nasym_ = nasym_.detach().cpu().numpy().reshape(-1)
        asym_list.extend(asym_)
        nasym_list.extend(nasym_)
        logging.info("Repeat: %d, Batch: %d, Mean Asym: %.6f" % (repeat, batch_id, np.mean(np.asarray(asym_list))))
        logging.info("Repeat: %d, Batch: %d, Mean NAsym: %.6f" % (repeat, batch_id, np.mean(np.asarray(nasym_list))))
      later = time.time()
      difference = int(later - now)
      logging.info("Time consumption: %d sec." % (difference))
      logging.info("Repeat %d || Noise level %.3f || Asym: %.6f." % (repeat, noise_levels[nl], np.mean(np.asarray(asym_list))))
      logging.info("Repeat %d || Noise level %.3f || NAsym: %.6f." % (repeat, noise_levels[nl], np.mean(np.asarray(nasym_list))))
      asym_nl.append(np.mean(np.asarray(asym_list)))
      nasym_nl.append(np.mean(np.asarray(nasym_list)))
    logging.info("Finish calculating Asym / NAsym.")

    # Draw and save symmetry distance.
    file_name = os.path.join(sym_dir, "repeat_{}".format(repeat))
    with tf.io.gfile.GFile(file_name+".npz", "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, asym=asym_nl, nasym=asym_nl)
      fout.write(io_buffer.getvalue())

def evaluate_asym(config, workdir):
  sym_dir = os.path.join(workdir, "asym")
  file_name = os.path.join(sym_dir, "repeat_{}".format(config.eval.repeat))
  asym = np.load(file_name+".npz")['asym']
  nasym = np.load(file_name+".npz")['nasym']
  logging.info("Asym: %.3e" % (np.mean(asym)))
  logging.info("NAsym: %.3e" % (np.mean(nasym)))

config_flags.DEFINE_config_file("config", None, "Configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_boolean("calculate", False, "Set as True to create encoded files.")
flags.DEFINE_boolean("read", False, "Set as True to display Asym.")
flags.DEFINE_integer("bs", 1000, "Batch size.")
flags.DEFINE_integer("slices", 1, "Number of random vectors used in Hutchinson's trace estimator (K).")
flags.DEFINE_integer("nl", 100, "Number of noise scales (T).")
flags.DEFINE_integer("total_repeat", 1, "Number of evaluations.")
flags.DEFINE_integer("checkpoint_idx", 70, "The index of a checkpoint file.")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
  workdir = os.path.join('results', FLAGS.workdir)
  config = FLAGS.config
  config.eval.batch_size = FLAGS.bs
  config.loss.slices = FLAGS.slices
  config.eval.noise_levels = FLAGS.nl
  config.eval.total_repeat = FLAGS.total_repeat
  config.sampling.score_restore_path = os.path.join(workdir, 'checkpoints/checkpoint_'+str(FLAGS.checkpoint_idx)+'.pth')
  tf.io.gfile.makedirs(workdir)
  if FLAGS.calculate:
    calculate_asym(config, workdir)
  if FLAGS.read:
    evaluate_asym(config, workdir)

if __name__ == "__main__":
  app.run(main)