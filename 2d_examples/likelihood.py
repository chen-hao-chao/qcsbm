"""
Likelihood function from: https://github.com/yang-song/score_sde_pytorch
"""
import torch
import numpy as np
from scipy import integrate

def from_flattened_numpy(x, shape):
  return torch.from_numpy(x.reshape(shape))

def to_flattened_numpy(x):
  return x.detach().cpu().numpy().reshape((-1,))

def get_div_fn(fn):
  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
  return div_fn

def get_likelihood_fn(sde, rtol=1e-3, atol=1e-5, method='RK45', eps=1e-5, is_score_model=False):
  def drift_and_div(model, x, t, noise):
    score_fn = model if is_score_model else model.score
    with torch.enable_grad():
      x.requires_grad_(True)
      rsde = sde.reverse(score_fn, probability_flow=True)
      drift = rsde.sde(x, t)[0]
      grad_fn_0 = torch.autograd.grad(drift[:,0].sum(), x, retain_graph=True)[0]
      grad_fn_1 = torch.autograd.grad(drift[:,1].sum(), x)[0]
    x.requires_grad_(False)
    div = grad_fn_0[:,0]+grad_fn_1[:,1]
    return drift, div

  def likelihood_fn(model, data):
    with torch.no_grad():
      shape = data.shape
      epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.

      def ode_func(t, x):
        sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift, div = drift_and_div(model, sample, vec_t, epsilon)
        drift = to_flattened_numpy(drift)
        logp_grad = to_flattened_numpy(div)
        return np.concatenate([drift, logp_grad], axis=0)

      init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      zp = solution.y[:, -1]
      z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
      prior_logp = sde.prior_logp(z)
      nll = -(prior_logp + delta_logp)
      
      return nll, z, nfe

  return likelihood_fn