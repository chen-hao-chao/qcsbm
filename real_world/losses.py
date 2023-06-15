"""All functions related to loss computation and optimization.
"""

from re import I
import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()
  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, balancing_fac=0.1, slices=16, energy=False):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """  
  # Stack the input K times.
  def to_sliced_tensor(batch, slices):
    sliced_batch = batch.unsqueeze(0).expand(slices, *batch.shape).contiguous().view(-1, batch.shape[1], batch.shape[2], batch.shape[3])
    return sliced_batch 
  # Stack the input K times.
  def to_sliced_vector(batch, slices):
    sliced_batch = batch.unsqueeze(0).expand(slices, *batch.shape).contiguous().view(-1)
    return sliced_batch

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    with torch.no_grad():
      score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous, energy=energy)
      t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
      ts = to_sliced_vector(t, slices)
      z = to_sliced_tensor(torch.randn_like(batch), slices)
      sliced_batch = to_sliced_tensor(batch, slices)
      mean, std = sde.marginal_prob(sliced_batch, ts)
      perturbed_data = mean + std[:, None, None, None] * z 
      perturbed_data = torch.tensor(perturbed_data, requires_grad=True)

    score = score_fn(perturbed_data, ts)
    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = 0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=1).view(slices, -1).mean(dim=0)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = 0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=1).view(slices, -1).mean(dim=0) * g2

    if balancing_fac > 0:
      # Sample random vectors from Rademacher distribution
      v = torch.randn_like(perturbed_data, device=batch.device).sign()
      # (1) Compute vJJ^Tv
      vs = torch.sum(score*v, dim=(1,2,3))
      gvs, = torch.autograd.grad(torch.sum(vs), perturbed_data, create_graph=True)
      traceJJt = torch.sum(torch.square(gvs), dim=(1,2,3))
      # (2) Compute vJJv
      gvs_detach = torch.tensor(gvs.detach().clone(), requires_grad=True)
      gvs_score = torch.sum(score * gvs_detach, dim=(1,2,3))
      gvss, = torch.autograd.grad(torch.sum(gvs_score), perturbed_data, create_graph=True)
      traceJJ = torch.sum(v*gvss, dim=(1,2,3))
      # (3) Calculate the regularization term and the total loss
      reg = ((traceJJt - traceJJ) * (std ** 2)).view(slices, -1).mean(dim=0)
      total_loss = torch.mean(losses + balancing_fac * reg)
      # (4) Perform backward propagation to compute the primary component
      total_loss.backward(retain_graph=True)
      # (5) Perform backward propagation to compute the secondary component
      gvs.backward(gvs_detach.grad)
    else:
      reg = torch.zeros(losses.shape)
      total_loss = torch.mean(losses)
      total_loss.backward()

    return torch.mean(losses), torch.mean(reg)
  
  return loss_fn

def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, balancing_fac=0.1, slices=5, energy=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting,
                              balancing_fac=balancing_fac, slices=slices, energy=energy)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    optimizer = state['optimizer']
    optimizer.zero_grad()
    loss, reg = loss_fn(model, batch)
    optimize_fn(optimizer, model.parameters(), step=state['step'])
    state['step'] += 1
    state['ema'].update(model.parameters())
    return loss, reg
  
  return step_fn
