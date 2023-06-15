from configs.default_toy_configs import get_default_configs
import torch

def get_config():
  config = get_default_configs()

  training = config.training
  training.n_iters = 10001

  optim = config.optim
  optim.lr = 1e-4
  
  config.seed = 8

  return config
