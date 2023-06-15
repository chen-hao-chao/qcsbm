from configs.default_toy_configs import get_default_configs

def get_config():
  config = get_default_configs()
  training = config.training
  training.n_iters = 2000

  data = config.data
  data.dataset = 'gaussian_8'

  loss = config.loss
  loss.loss_type = 'dsm'
  loss.reg_type = 'lqc'
  loss.balancing_fac = 0.1
  loss.slices = 1

  model = config.model
  model.act_ = 'swish'
  model.act__ = 'lrelu'

  return config
