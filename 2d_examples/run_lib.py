import os
import logging

import datasets
import losses
from models import simple_score_fn
from utils import plot_vector_field, get_oracle_score_pointwise, calculate_score_error, plot_data_points
import sampling_fn
import torch
import tensorflow as tf
import torch.optim as optim
from torch.utils import tensorboard
import datetime
import likelihood
import sde_lib
from prdc import compute_prdc

def run(config, workdir):
  """Start a training process for SBMs.
  Args:
    config: (dict) Experimental configuration file that specifies the hyper-parameters.
    workdir: (str) Working directory for saving the checkpoints and Tensorflow summaries.
  """
  # Create directories for experimental logs.
  visualization_dir = os.path.join(workdir, "visualization")
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(visualization_dir)
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(tb_dir)

  # Create Tensorboard files.
  loc_dt_format = datetime.datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
  tb_current_file_dir = os.path.join(tb_dir, loc_dt_format)
  tf.io.gfile.makedirs(tb_current_file_dir)
  writer = tensorboard.SummaryWriter(tb_current_file_dir)

  # Initialize an SBM.
  score_model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device, dtype=config.data.dtype)

  # Get the function for performing training steps.
  train_step_fn = losses.get_step_fn(reg_type=config.loss.reg_type,
                                     std_value_max=config.loss.std_max, std_value_min=config.loss.std_min,
                                     balancing_fac=config.loss.balancing_fac, slices=config.loss.slices,
                                     dtype=config.data.dtype)
  
  # Initialize an optimizer.
  optimizer = optim.Adam(score_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  
  # Get the function for calulating the likelihood.
  sde = sde_lib.VESDE(sigma_min=config.loss.std_min, sigma_max=config.loss.std_max)
  likelihood_fn = likelihood.get_likelihood_fn(sde=sde, rtol=1e-3, atol=1e-5, is_score_model=(config.model.type=='score_model'))

  # Get a mini-batch of testing data for evaluation.
  bs = config.training.batch_size
  config.training.batch_size = 5000
  _ds = datasets.get_dataset(config)
  iter_ds = iter(_ds)
  points = torch.tensor(next(iter_ds)['position']._numpy()).to(config.device, dtype=config.data.dtype)
  config.training.batch_size = bs

  # Get oracle scores. 
  logging.info("Calculating the oracle score...")
  score_oracle, _, noise_points = get_oracle_score_pointwise(config, points=points, ds=config.data.dataset)
  noise_points = torch.tensor(noise_points).to(config.device, dtype=config.data.dtype)

  # Get a sampler.
  shape = (config.training.batch_size, 2)
  sampler = sampling_fn.get_ode_sampler(sde, shape, rtol=1e-3, atol=1e-5,
                    method='RK45', eps=1e-5, device='cuda', is_score_model=(config.model.type=='score_model'))
  
  # Build an iterator.
  ds = datasets.get_dataset(config)
  iter_ds = iter(ds)

  # Start training.
  logging.info("Start training.")
  for step in range(config.training.n_iters + 1):
    # Execute one training step.
    data = next(iter_ds)
    batch = torch.from_numpy(data['position']._numpy()).to(config.device, dtype=config.data.dtype)
    loss, reg = train_step_fn(score_model, optimizer, batch, config.model.type=='score_model')

    # Print the loss periodically.
    if step % config.training.log_freq == 0:
      expected_score_error = calculate_score_error(noise_points, score_oracle, score_model, is_score_model=(config.model.type=='score_model'), n_scale=config.model.n_scales)
      nll, _, _ = likelihood_fn(score_model, points)
      logging.info("step: %d, loss_sm: %.6e" % (step, loss.item()))
      logging.info("step: %d, loss_reg: %.6e" % (step, reg.item()))
      logging.info("step: %d, expected_score_error: %.6e" % (step, expected_score_error))
      logging.info("step: %d, NLL: %.6e" % (step, nll.mean().item()))
      logging.info("----")
      writer.add_scalar("loss_sm", loss, step)
      writer.add_scalar("loss_reg", reg, step)
      writer.add_scalar("expected_score_error", expected_score_error, step)
      writer.add_scalar("NLL", nll.mean(), step)

    # Save a checkpoint periodically.
    if step != 0 and step % config.training.snapshot_freq == 0:
      save_step = step // config.training.snapshot_freq
      torch.save({'model': score_model.state_dict(),}, os.path.join(checkpoint_dir, 'checkpoint_{}.pth'.format(step)))
    
    # Evaluate the sampling performance and visualize the vector fields periodically.
    if step != 0 and step % config.training.plot_freq == 0:
      # Perform sampling and calculate precision / recall.
      save_step = step // config.training.plot_freq
      samples, _ = sampler(score_model)
      samples = samples.cpu().detach().numpy()
      prdc = compute_prdc(real_features=batch.cpu().detach().numpy(),
                            fake_features=samples,
                            nearest_k=config.eval.nearest_k)
      logging.info("step: %d, precision: %.6e" % (step, prdc['precision']))
      logging.info("step: %d, recall: %.6e" % (step, prdc['recall']))
      writer.add_scalar("precision", prdc['precision'], step)
      writer.add_scalar("recall", prdc['recall'], step)

      # Plot the sampled points and the output vector field of the SBM.
      plot_data_points(config, samples, os.path.join(visualization_dir, "vf_"+str(save_step)+".png"))
      plot_vector_field(config, score_model, os.path.join(visualization_dir, "vf_"+str(save_step)+".png"))