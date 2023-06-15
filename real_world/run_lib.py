import os
import numpy as np
import tensorflow as tf
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, ncsnpp_mod
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint, evaluate_nll

FLAGS = flags.FLAGS

def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  if config.training.restore_path:
    logging.info("Checkpoint restored from: %s" % config.training.restore_path)
    state = restore_checkpoint(config.training.restore_path, state, config.device)
  else:
    logging.info("Initialize a new score model.")
  initial_step = config.training.init_step

  # Build data iterators
  train_ds, _, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)
  
  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  _, ds_bpd, _ = datasets.get_dataset(config, uniform_dequantization=True, evaluation=True)
  ds_bpd_iter = iter(ds_bpd)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  balancing_fac = config.loss.balancing_fac
  slices = config.loss.slices
  energy = config.model.energy
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting,
                                     balancing_fac=balancing_fac, slices=slices,
                                     energy=energy)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.sampling.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  
  # Build the likelihood computation function when likelihood is enabled
  likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, energy=energy)
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    # Execute one training step
    loss, reg = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)
      logging.info("step: %d, training_reg: %.5e" % (step, reg.item()))
      writer.add_scalar("training_reg", reg, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Save a checkpoint and generate samples periodically
    if step != 0 and step % config.training.snapshot_freq == 0:
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      logging.info("Calculating NLL...")
      try:
        bpd_batch = next(ds_bpd_iter)
      except:
        ds_bpd_iter = iter(ds_bpd)
        bpd_batch = next(ds_bpd_iter)
      nll = evaluate_nll(config, score_model, scaler, bpd_batch, likelihood_fn)
      logging.info("step: %d, nll: %.5e bit/dim" % (step, nll))
      writer.add_scalar("nll", nll, step)
      
      logging.info("Generating samples...")
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          save_image(image_grid, fout)