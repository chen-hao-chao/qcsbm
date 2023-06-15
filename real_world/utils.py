import torch
import logging
import numpy as np
import time
from models import utils as mutils

def restore_checkpoint(ckpt_dir, state, device):
  loaded_state = torch.load(ckpt_dir, map_location=device)
  state['optimizer'].load_state_dict(loaded_state['optimizer'])
  state['model'].load_state_dict(loaded_state['model'], strict=False)
  state['ema'].load_state_dict(loaded_state['ema'])
  return state

def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
  }
  torch.save(saved_state, ckpt_dir)

def evaluate_nll(config, score_model, scaler, batch, likelihood_fn):
  score_model.eval()
  now = time.time()
  eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
  eval_batch = eval_batch.permute(0, 3, 1, 2)
  eval_batch = scaler(eval_batch)
  bpd = likelihood_fn(score_model, eval_batch)[0]
  bpd = bpd.detach().cpu().numpy().reshape(-1)
  later = time.time()
  difference = int(later - now)
  logging.info("Time consumption: %d sec." % (difference))
  return np.mean(bpd)

def get_sym_function(sde, continuous, energy, slices=1, eps=1e-5):
  def to_sliced_tensor(batch, slices):
    sliced_batch = batch.unsqueeze(0).expand(slices, *batch.shape).contiguous().view(-1, batch.shape[1], batch.shape[2], batch.shape[3])
    return sliced_batch 
  def to_sliced_vector(batch, slices):
    sliced_batch = batch.unsqueeze(0).expand(slices, *batch.shape).contiguous().view(-1)
    return sliced_batch
  def evaluate_sym(score_model, batch, noise_level):
    score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous, energy=energy)
    t = (torch.ones(batch.shape[0], device=batch.device) * noise_level) * (sde.T - eps) + eps
    ts = to_sliced_vector(t, slices)
    z = to_sliced_tensor(torch.randn_like(batch), slices) 
    sliced_batch = to_sliced_tensor(batch, slices)
    mean, std = sde.marginal_prob(sliced_batch, ts)
    perturbed_data = mean + std[:, None, None, None] * z 
    perturbed_data = torch.tensor(perturbed_data, requires_grad=True) 
    score = score_fn(perturbed_data, ts)
    v = torch.randn_like(batch, device=batch.device).sign()
    vs = torch.sum(score*v, dim=(1,2,3))
    gvs, = torch.autograd.grad(torch.sum(vs), perturbed_data, retain_graph=True)
    gvs_detach = torch.tensor(gvs.detach().clone(), requires_grad=True)
    gvs_score = torch.sum(score * gvs_detach, dim=(1,2,3))
    gvss, = torch.autograd.grad(torch.sum(gvs_score), perturbed_data)
    traceJJ = torch.sum(v*gvss, dim=(1,2,3))
    traceJJt = torch.sum(torch.square(gvs), dim=(1,2,3))
    squared_asym_norm = (0.5 * (traceJJt - traceJJ))
    squared_asym = (squared_asym_norm / (traceJJt+1e-8)).view(slices, -1).mean(dim=0)
    reg = (traceJJt - traceJJ).view(slices, -1).mean(dim=0)
    return reg, squared_asym
  return evaluate_sym