import torch
from utils import calculate_asymmetricity, get_oracle_score_pointwise, calculate_score_error
from models import simple_score_fn
import likelihood
import datasets
import logging

import sampling_fn
import sde_lib
from prdc import compute_prdc

def eval_sampling(config):
  # Load checkpoint.
  score_model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
  try:
    checkpoint = torch.load(config.eval.restore_path, map_location=config.device)
    score_model.load_state_dict(checkpoint['model'], strict=False)
  except:
    raise ValueError("Checkpoint {} cannot be found.".format(config.eval.restore_path))

  # Get testing samples.
  config.training.batch_size = 5000
  _ds = datasets.get_dataset(config)
  iter_ds = iter(_ds)
  batch_eval = torch.tensor(next(iter_ds)['position']._numpy()).to(config.device, dtype=config.data.dtype)
  
  # Create a sampler.
  sde = sde_lib.VESDE(sigma_min=config.loss.std_min, sigma_max=config.loss.std_max)
  shape = (config.training.batch_size, 2)
  sampler = sampling_fn.get_ode_sampler(sde, shape, rtol=1e-3, atol=1e-5,
                  method='RK45', eps=1e-5, device='cuda', is_score_model=(config.model.type=='score_model'))

  # Sample data points and calculate precision and recall.
  samples, _ = sampler(score_model)
  samples = samples.cpu().detach().numpy()
  prdc = compute_prdc(real_features=batch_eval.cpu().detach().numpy(),
                        fake_features=samples,
                        nearest_k=config.eval.nearest_k)
  logging.info("precision: %.6e" % (prdc['precision']))
  logging.info("recall: %.6e" % (prdc['recall']))
  
def eval_sm_err(config):
  # Load checkpoint.
  score_model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
  try:
    checkpoint = torch.load(config.eval.restore_path, map_location=config.device)
    score_model.load_state_dict(checkpoint['model'], strict=False)
  except:
    raise ValueError("Checkpoint {} cannot be found.".format(config.eval.restore_path))
  
  # Get testing samples.
  config.training.batch_size = 5000
  _ds = datasets.get_dataset(config)
  iter_ds = iter(_ds)
  batch_eval = torch.tensor(next(iter_ds)['position']._numpy()).to(config.device, dtype=config.data.dtype)

  # Get oracle scores.
  score_oracle, _, noise_points = get_oracle_score_pointwise(config, points=batch_eval, ds=config.data.dataset)
  noise_points = torch.tensor(noise_points).to(config.device, dtype=config.data.dtype)

  # Calculate score errors, Asym, and NAsym.
  expected_score_error = calculate_score_error(noise_points, score_oracle, score_model,
                                              is_score_model=(config.model.type=='score_model'),
                                              n_scale=config.model.n_scales)
  nasym_expected_mean, asym_expected_mean = calculate_asymmetricity(noise_points, score_model,
                                              is_score_model=(config.model.type=='score_model'),
                                              n_scale=config.model.n_scales)
  logging.info("Expected score error: %.6e" % (expected_score_error))
  logging.info("Expected nasym: %.6e" % (nasym_expected_mean))
  logging.info("Expected asym: %.6e" % (asym_expected_mean))

def eval_nll(config):
  # Load checkpoint.
  score_model = simple_score_fn.simple_noise_conditioned_score_fn(config).to(config.device)
  try:
    checkpoint = torch.load(config.eval.restore_path, map_location=config.device)
    score_model.load_state_dict(checkpoint['model'], strict=False)
  except:
    raise ValueError("Checkpoint {} cannot be found.".format(config.eval.restore_path))

  # Get an eval batch
  config.training.batch_size = config.data.dataset_size
  split_size = 10000
  _ds = datasets.get_dataset(config)
  iter_ds = iter(_ds)
  batch_eval = torch.tensor(next(iter_ds)['position']._numpy()).to(config.device, dtype=config.data.dtype)
  
  # Setup likelihood function.
  sde = sde_lib.VESDE(sigma_min=config.loss.std_min, sigma_max=config.loss.std_max) #, N=config.model.n_scales
  likelihood_fn = likelihood.get_likelihood_fn(sde=sde, rtol=1e-3, atol=1e-5, is_score_model=(config.model.type=='score_model'))
  
  # Calculate NLL.
  splits = config.training.batch_size // split_size
  avg_nll = 0
  avg_nfe = 0
  for i in range(splits):
    batch_eval_split = torch.tensor(batch_eval[i*split_size:(i+1)*split_size]).to(config.device)
    nll, _, nfe = likelihood_fn(score_model, batch_eval_split)
    avg_nll += nll.mean().item() / splits
    avg_nfe += nfe / splits
  logging.info("NLL: %.6e" % (avg_nll))
  logging.info("NFE: %d" % (avg_nfe))

def run(config):
  if config.eval.type == "nll":
    eval_nll(config)
  elif config.eval.type == "score_err":
    eval_sm_err(config)
  elif config.eval.type == "sampling":
    eval_sampling(config)
  else:
    raise NotImplementedError("Evaluation type {} unknown.".format(config.eval.type))
