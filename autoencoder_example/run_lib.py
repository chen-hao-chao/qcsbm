import os
import logging
import numpy as np

from models import simple_untied_autoencoder

import torch
import torch.nn
import tensorflow as tf
import torch.optim as optim
from torch.utils import tensorboard
import datetime

def run(config, workdir):
  # Create directories for experimental logs.
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(tb_dir)

  # Tensorboard
  loc_dt_format = datetime.datetime.today().strftime("_%Y-%m-%d_%H:%M:%S")
  tb_current_file_dir = os.path.join(tb_dir, "seed_"+str(config.seed)+loc_dt_format)
  tf.io.gfile.makedirs(tb_current_file_dir)
  writer = tensorboard.SummaryWriter(tb_current_file_dir)

  # Initialize the score model.
  score_model = simple_untied_autoencoder.simple_untied_autoencoder().to(config.device, dtype=config.data.dtype)
  
  # Initialize the optimizer.
  optimizer = optim.Adam(score_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)

  # Get points
  w = config.sampling.width
  h = config.sampling.height
  d = config.sampling.density
  x, y = np.meshgrid(np.linspace(-w, w, d, dtype=np.float32), np.linspace(-h, h, d, dtype=np.float32))
  points = torch.tensor(np.concatenate((np.expand_dims(x.flatten(), axis=1), np.expand_dims(y.flatten(), axis=1)), axis=1)).to(config.device, dtype=config.data.dtype)
  
  # Start training.
  logging.info("Start training.")
  for step in range(config.training.n_iters + 1):
    # Execute one training step.
    optimizer.zero_grad()
    score_model.train()
    batch = torch.tensor(points, requires_grad=True)
    score = score_model(batch)
    grad0, = torch.autograd.grad(torch.sum(score[torch.arange(score.shape[0]), 0]), batch, create_graph=True)
    grad1, = torch.autograd.grad(torch.sum(score[torch.arange(score.shape[0]), 1]), batch, create_graph=True)
    traceJJt_true = torch.sum(torch.square(grad0), dim=1) + torch.sum(torch.square(grad1), dim=1) 
    traceJJ_true = torch.square(grad0[:,0]) + torch.square(grad1[:,1]) + 2*grad0[:,1]*grad1[:,0] 
    reg = torch.mean(traceJJt_true - traceJJ_true)
    reg.backward()
    torch.nn.utils.clip_grad_norm_(score_model.parameters(), 1.0)
    optimizer.step()

    # Print the loss periodically.
    if step % config.training.log_freq == 0:
      r0 = score_model.R.weight[0,:]
      r1 = score_model.R.weight[1,:]
      w0 = score_model.Wt.weight.t()[0,:]
      w1 = score_model.Wt.weight.t()[1,:]
      with torch.no_grad():
        sym_diff = torch.abs(torch.sum(r0*w1) - torch.sum(r1*w0))
        weight_diff = torch.sqrt(torch.sum(torch.square(score_model.R.weight - score_model.Wt.weight.t())))
      logging.info("step: %d, loss_reg: %.6e" % (step, reg.item()))
      logging.info("step: %d, sym_diff: %.6e" % (step, sym_diff.item()))
      logging.info("step: %d, weight_diff: %.6e" % (step, weight_diff.item()))
      logging.info("----")
      writer.add_scalar("loss_reg", reg, step)
      writer.add_scalar("sym_diff", sym_diff, step)
      writer.add_scalar("weight_diff", weight_diff, step)
    
  # Save a checkpoint periodically.
  torch.save({'model': score_model.state_dict(),}, os.path.join(checkpoint_dir, 'checkpoint_seed_{}.pth'.format(config.seed)))