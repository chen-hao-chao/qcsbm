import torch
import torch.nn as nn

class Swish(nn.Module):
  def __init__(self, dim=-1):
    """
    Swish from: https://github.com/wgrathwohl/LSD/blob/master/networks.py#L299
    """
    super().__init__()
    if dim > 0:
      self.beta = nn.Parameter(torch.ones((dim,)))
    else:
      self.beta = torch.ones((1,))

  def forward(self, x):
    if len(x.size()) == 2:
      return x * torch.sigmoid(self.beta[None, :] * x)
    else:
      return x * torch.sigmoid(self.beta[None, :, None, None] * x)

class simple_noise_conditioned_score_fn(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.concat_network = nn.Linear(256 + 1, 2)
    self.act_ = nn.ELU() if config.model.act_ == 'elu' else (nn.LeakyReLU(0.1) if config.model.act_ == 'lrelu' else Swish(256 + 1)) 
    self.act__ = nn.ELU() if config.model.act__ == 'elu' else (nn.LeakyReLU(0.1) if config.model.act__ == 'lrelu' else Swish(2))
    self.input_network = nn.Sequential(
      nn.Linear(2, 64), Swish(64),
      nn.Linear(64, 128), Swish(128),
      nn.Linear(128, 256)
    )
    self.noise_network = nn.Sequential(
      nn.Linear(1, 1), Swish(1),
      nn.Linear(1, 1), Swish(1),
      nn.Linear(1, 1)
    )

  def forward(self, x, n_cond):
    n_cond = (self.config.loss.std_min * (self.config.loss.std_max / self.config.loss.std_min) ** n_cond)
    n_cond_ = torch.unsqueeze(n_cond, 1)
    n_cond_ = self.noise_network(n_cond_)
    x_ = self.input_network(x)
    x_cond_cat = torch.cat((x_, n_cond_), 1)
    out = (x - self.concat_network(self.act_(x_cond_cat))) / (n_cond[:,None])
    return out

  def energy(self, x, n_cond):
    n_cond = (self.config.loss.std_min * (self.config.loss.std_max / self.config.loss.std_min) ** n_cond)
    n_cond_ = torch.unsqueeze(n_cond, 1)
    n_cond_ = self.noise_network(n_cond_)
    x = x.requires_grad_()
    x_ = self.input_network(x)
    x_cond_cat = torch.cat((x_, n_cond_), 1)
    out = self.act__(self.concat_network(self.act_(x_cond_cat)))
    energy = torch.sum(torch.square(x - out), dim=1) / (n_cond * 2)
    return energy
    
  def score(self, x, n_cond):
    n_cond = (self.config.loss.std_min * (self.config.loss.std_max / self.config.loss.std_min) ** n_cond)
    n_cond_ = torch.unsqueeze(n_cond, 1)
    n_cond_ = self.noise_network(n_cond_)
    with torch.enable_grad():
      x = x.requires_grad_()
      x_ = self.input_network(x)
      x_cond_cat = torch.cat((x_, n_cond_), 1)
      out = self.act__(self.concat_network(self.act_(x_cond_cat)))
      energy = torch.sum(torch.square(x - out), dim=1) / (n_cond * 2)
      score = - torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
    return score
  
  def score_inference(self, x, n_cond):
    n_cond = (self.config.loss.std_min * (self.config.loss.std_max / self.config.loss.std_min) ** n_cond)
    n_cond_ = torch.unsqueeze(n_cond, 1)
    n_cond_ = self.noise_network(n_cond_)
    with torch.enable_grad():
      x = x.requires_grad_()
      x_ = self.input_network(x)
      x_cond_cat = torch.cat((x_, n_cond_), 1)
      out = self.act__(self.concat_network(self.act_(x_cond_cat)))
      energy = torch.sum(torch.square(x - out), dim=1) / (n_cond * 2)
    return - torch.autograd.grad(energy.sum(), x)[0]