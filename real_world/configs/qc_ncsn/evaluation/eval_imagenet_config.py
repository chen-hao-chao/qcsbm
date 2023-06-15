"""Training NCSN++ on ImageNet with VE SDE."""

from configs.default_imagenet_configs import get_default_configs

 
def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.n_iters = 750001

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.score_restore_path = ""

  # ($) evaluate
  eval = config.eval
  eval.batch_size = 256

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.fourier_scale = 16
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.0
  model.conv_size = 3
  
  eval = config.eval
  eval.batch_size = 100
  eval.num_samples = 50000
  eval.division_idx = 0
  eval.num_divisions = 10
  eval.encoding_rounds = 50
  eval.noise_levels = 100
  eval.repeat = 0
  eval.total_repeat = 1
  eval.nearest_k = 3

  return config
