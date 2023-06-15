import torch
import torch.nn as nn

class simple_untied_autoencoder(nn.Module):
  """simple score model"""
  def __init__(self):
    """ Output = R Sigmoid( W^T Input + b ) + c """
    super().__init__()
    self.R = nn.Linear(2, 2)
    self.Wt = nn.Linear(2, 2)
    self.Sig = Swish(2)
    self.param = 1.0
    # Initialize with uniform weights between [-1,1]
    self.R.weight.data.uniform_(-self.param, self.param)
    self.R.bias.data.uniform_(-self.param, self.param)
    self.Wt.weight.data.uniform_(-self.param, self.param)
    self.Wt.bias.data.uniform_(-self.param, self.param)

  def forward(self, x):
    return self.R(self.Sig(self.Wt(x)))

class Swish(nn.Module):
  def __init__(self, dim=-1):
    """
    Swish from https://github.com/wgrathwohl/LSD/blob/master/networks.py#L299
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