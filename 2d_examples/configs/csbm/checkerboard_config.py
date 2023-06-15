from configs.default_toy_configs import get_default_configs

def get_config():
  config = get_default_configs()
  training = config.training
  training.n_iters = 4000
  
  data = config.data
  data.dataset = 'checkerboard'

  loss = config.loss
  loss.loss_type = 'dsm'
  loss.reg_type = 'none'
  loss.balancing_fac = 0.0
  loss.slices = 1

  model = config.model
  model.type = 'energy_model'
  model.act_ = 'swish'
  model.act__ = 'elu'

  return config
