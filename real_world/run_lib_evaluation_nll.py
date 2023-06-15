import logging
import numpy as np
import os
import io
import glob
import time
import tensorflow as tf
import torch
from ml_collections.config_flags import config_flags
from absl import flags
from absl import app

import losses
import datasets
import sde_lib
import likelihood
from models import ncsnpp, ncsnpp_mod
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint

FLAGS = flags.FLAGS

def calculate_nll(config, workdir):
  repeat = config.eval.repeat
  # Create folder
  nll_dir = os.path.join(workdir, "nll")
  tf.io.gfile.makedirs(nll_dir)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  state = restore_checkpoint(config.sampling.score_restore_path, state, config.device) 

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the likelihood computation function when likelihood is enabled
  likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, atol=FLAGS.atol, rtol=FLAGS.rtol, energy=config.model.energy) #, atol=FLAGS.atol, rtol=FLAGS.rtol
  bpds = []
  nfes = []
  
  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  _, ds_bpd, _ = datasets.get_dataset(config, uniform_dequantization=True, evaluation=True)
  bpd_iter = iter(ds_bpd)
  iter_num = len(ds_bpd)
  for repeat in range(config.eval.total_repeat):
    for batch_id in range(iter_num):
      logging.info("Evaluating the images [%d, %d]..." % (batch_id*config.eval.batch_size, (batch_id+1)*config.eval.batch_size))
      now = time.time()
      batch = next(bpd_iter)        
      eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      bpd, _, nfe = likelihood_fn(score_model, eval_batch)
      bpd = bpd.detach().cpu().numpy().reshape(-1)
      bpds.extend(bpd)
      nfes.append(nfe)
      logging.info("Repeat: %d, Batch: %d, Mean NLL: %.3f" % (repeat, batch_id, np.mean(np.asarray(bpds))))
      logging.info("Repeat: %d, Batch: %d, Mean NFE: %d" % (repeat, batch_id, np.mean(np.asarray(nfes))))
      with tf.io.gfile.GFile(os.path.join(nll_dir, "bpd_{}.npz".format(batch_id)), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, bpd=bpd, nfe=nfe)
        fout.write(io_buffer.getvalue())
      later = time.time()
      difference = int(later - now)
      logging.info("Time consumption: %d sec." % (difference))
  logging.info("NLL: %.3f bit/dim." % (np.mean(np.asarray(bpds))))
  logging.info("NFE: %d" % (np.mean(np.asarray(nfes))))
  logging.info("Finish calculating NLL.")

def evaluate_nll(config, workdir):
  nll_dir = os.path.join(workdir, "nll") 
  bpds = []
  nfes = []
  for repeat in range(config.eval.total_repeat):
    for file in glob.glob(os.path.join(nll_dir, '*.npz')):
      f = np.load(file)
      bpd = f['bpd']
      bpds.extend(bpd)
      nfe = f['nfe']
      nfes.append(nfe)
  logging.info("Repeat %d || NLL: %.3f bit/dim." % (repeat, np.mean(np.asarray(bpds))))
  logging.info("Repeat %d || NFE: %d." % (repeat, np.mean(np.asarray(nfes))))
  logging.info("Finish calculating NLL.")

config_flags.DEFINE_config_file("config", None, "Configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_boolean("calculate", False, "Set as True to create encoded files.")
flags.DEFINE_boolean("read", False, "Set as True to calculate NLL.")
flags.DEFINE_integer("bs", 1000, "Batch size.")
flags.DEFINE_integer("total_repeat", 1, "Number of evaluations.")
flags.DEFINE_integer("checkpoint_idx", 70, "The index of a checkpoint file.")
flags.DEFINE_float("atol", 0.00001, "The atol value of an ODE solver.")
flags.DEFINE_float("rtol", 0.001, "The rtol value of an ODE solver.")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
  workdir = os.path.join('results', FLAGS.workdir)
  config = FLAGS.config
  config.eval.batch_size = FLAGS.bs
  config.eval.total_repeat = FLAGS.total_repeat
  config.sampling.score_restore_path = os.path.join(workdir, 'checkpoints/checkpoint_'+str(FLAGS.checkpoint_idx)+'.pth')
  tf.io.gfile.makedirs(workdir)
  if FLAGS.calculate:
    calculate_nll(config, workdir)
  if FLAGS.read:
    evaluate_nll(config, workdir)

if __name__ == "__main__":
  app.run(main)