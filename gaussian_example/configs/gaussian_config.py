import ml_collections
import torch

def get_config():
  config = ml_collections.ConfigDict()
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.width = 3
  sampling.height = 3
  sampling.density = 20
  sampling.shape = (1000, 2)
  sampling.steps = 300
  sampling.num_traj = 10
  sampling.init_variance = 0.5
  sampling.init_position = -2
  sampling.std = 0.1
  sampling.type = 'usbm'
  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config
