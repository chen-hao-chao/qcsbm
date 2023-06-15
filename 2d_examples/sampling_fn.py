"""
Sampling function from: https://github.com/yang-song/score_sde_pytorch
"""
import abc
import torch

from scipy import integrate

def from_flattened_numpy(x, shape):
  return torch.from_numpy(x.reshape(shape))

def to_flattened_numpy(x):
  return x.detach().cpu().numpy().reshape((-1,))

class Predictor(abc.ABC):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    pass

class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None] * z
    return x, x_mean

def get_ode_sampler(sde, shape, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda', is_score_model=True):
  def drift_fn(model, x, t):
    score_fn = model if is_score_model else model.score
    with torch.enable_grad():
      x.requires_grad_(True)
      rsde = sde.reverse(score_fn, probability_flow=True)
      drift = rsde.sde(x, t)[0]
    return drift
  
  def denoise_update_fn(model, x):
    score_fn = model if is_score_model else model.score
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def ode_sampler(model, z=None):
    with torch.no_grad():
      if z is None:
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)
      
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
      x = denoise_update_fn(model, x)
      return x, nfe

  return ode_sampler