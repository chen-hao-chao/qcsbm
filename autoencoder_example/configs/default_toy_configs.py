import ml_collections
import torch

def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.n_iters = 20000
  training.batch_size = 5000
  training.snapshot_freq = 500 
  training.log_freq = 200
  training.plot_freq = 1000
  training.deterministic = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.type = 'ode'
  sampling.noise_std = 1.0
  sampling.num_steps = 1000
  sampling.epsilon = 0.01
  sampling.width = 10
  sampling.height = 10
  sampling.density = 40
  sampling.shape = (1000, 2)

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'rings'
  data.dtype = torch.float32
  data.dataset_size = 50000
  data.shuffle_files = True

  # model
  config.model = model = ml_collections.ConfigDict()
  model.type = 'score_model'
  model.nf = 16
  model.nf1 = 128
  model.nf2 = 64
  model.nf3 = 128
  model.noise_conditioned = False
  model.n_scales = 10
  model.sf = 1.0
  model.act = 'elu'

  # loss
  config.loss = loss = ml_collections.ConfigDict()
  loss.loss_type = 'dsm'
  loss.reg_type = 'l2-est'
  loss.std = 0.1
  loss.std_max = 3.0 #5.0
  loss.std_min = 0.1 #0.5
  loss.balancing_fac = 0.1
  loss.slices = 1

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.lr = 7.5e-4 #1e-2
  optim.beta1 = 0.9
  optim.eps = 1e-8

  # Evaluation
  config.eval = eval = ml_collections.ConfigDict()
  eval.type = "sampling"

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config