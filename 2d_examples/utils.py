import logging
import numpy as np

import torch
from torch.autograd import Variable

import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import datasets

matplotlib.use('Agg')

def oracle_score_denominator(point, batch, sigma):
  return torch.exp(  - ( ( (batch[:, 0]-point[0])**2 + (batch[:, 1]-point[1])**2 )  / (2*(sigma**2)) )   )  /  ( 2*np.pi*(sigma**2) )

def guassian_prob(x, tx, sigma):
  return torch.exp(  - ( ( (tx[:, 0]-x[0])**2 + (tx[:, 1]-x[1])**2 )  / (2*(sigma**2)) )   )  /  ( 2*np.pi*(sigma**2) )  

def grad_log_guassian_prob(x, tx, sigma):
  return (x - tx) /  (sigma**2)  

def oracle_score_numerator(point, batch, sigma):
  p = guassian_prob(point, batch, sigma)
  diff = ( batch-point ) /  (sigma**2)
  diff[:, 0] *= p
  diff[:, 1] *= p
  return diff, p

def get_oracle_score(points, batch, sigma, eps=1e-8):
  points = torch.tensor(points)
  numerator, prob = oracle_score_numerator(points, batch, sigma = sigma)
  denominator = oracle_score_denominator(points, batch, sigma = sigma)
  sum_numerator = torch.sum(numerator, dim=0)
  sum_denominator = torch.sum(denominator, dim=0) 
  prob = torch.mean(prob, dim=0)
  return sum_numerator / (sum_denominator + eps), prob

def get_oracle_score_pointwise(config, points, ds):
  bs = config.training.batch_size
  config.data.dataset = ds
  config.training.batch_size = int(config.data.dataset_size)
  _ds = datasets.get_dataset(config)
  iter_ds = iter(_ds)
  batch = torch.tensor(next(iter_ds)['position']._numpy()).to(config.device)

  n_scale = config.model.n_scales
  score_oracle = np.zeros((n_scale+1, points.shape[0], points.shape[1]))
  prob_oracle = np.zeros((n_scale+1, points.shape[0]))
  noise_points = np.zeros((n_scale+1, points.shape[0], points.shape[1]))

  for ns in range(n_scale+1):
    timestep = (ns) / (n_scale+1)
    std = (config.loss.std_min * (config.loss.std_max / config.loss.std_min) ** timestep)
    
    z = torch.randn_like(points, device=points.device)
    z_ = torch.randn_like(batch, device=batch.device)
    
    noise_pts = points + std * z
    noise_bat = batch + std * z_
      
    for i in range(points.shape[0]):
      score_pt, prob_pt = get_oracle_score(noise_pts[i], noise_bat, std)
      score_oracle[ns, i, :] = score_pt.cpu().numpy()
      prob_oracle[ns, i] = prob_pt.cpu().numpy()
      noise_points[ns, i, :] = noise_pts[i].cpu().numpy()
  
  config.training.batch_size = bs 
  return score_oracle, prob_oracle, noise_points

def calculate_score_error(noise_points, score_oracle, score_model, is_score_model=True, n_scale=10):
  score_model.eval()
  avg_norm = 0
  for ns in range(n_scale):
    points = Variable(noise_points[ns], requires_grad=True)
    timestep = (ns+1) / (n_scale)
    t = torch.ones(points.shape[0], device=points.device) * timestep
    if is_score_model:
      score_estimate = score_model(points, t)
      score_estimate_np = score_estimate.clone().detach().cpu().numpy()
      norm = (score_oracle[ns,:,0] - score_estimate_np[:,0])**2 + (score_oracle[ns,:,1] - score_estimate_np[:,1])**2
    else:
      score_estimate = score_model.score_inference(points, t)
      score_estimate_np = score_estimate.clone().detach().cpu().numpy()
      norm = (score_oracle[ns,:,0] - score_estimate_np[:,0])**2 + (score_oracle[ns,:,1] - score_estimate_np[:,1])**2
    
    avg_norm += np.mean(0.5*norm) / (n_scale)

  return avg_norm

def calculate_asymmetricity(noise_points, score_model, is_score_model=True, n_scale=10):
  avg_asym = 0
  avg_nasym = 0
  
  for ns in range(n_scale):
    points = Variable(noise_points[ns], requires_grad=True)
    timestep = (ns+1) / (n_scale)
    t = torch.ones(points.shape[0], device=points.device) * timestep
    if is_score_model:
      score_estimate = score_model(points, t)
    else:
      score_estimate = score_model.score(points, t)
    grad0, = torch.autograd.grad(torch.sum(score_estimate[torch.arange(score_estimate.shape[0]), 0]), points, retain_graph=True)
    grad1, = torch.autograd.grad(torch.sum(score_estimate[torch.arange(score_estimate.shape[0]), 1]), points)
    traceJJt = torch.sum(torch.square(grad0), dim=1) + torch.sum(torch.square(grad1), dim=1)
    traceJJ = torch.square(grad0[:,0]) + torch.square(grad1[:,1]) + 2*grad0[:,1]*grad1[:,0]
    squared_asym_norm = 0.5 * (traceJJt - traceJJ)
    nasym = (squared_asym_norm / (traceJJt+1e-8)).cpu().numpy()
    asym = (traceJJt - traceJJ).cpu().numpy()

    avg_asym += np.mean(asym) / (n_scale)
    avg_nasym += np.mean(nasym) / (n_scale)

  return avg_nasym, avg_asym
  
def plot_vector_field(config, score_fn, dir_file):
  logging.info("Plotting Vector Field...")
  w = config.sampling.width
  h = config.sampling.height
  density = config.sampling.density
  with torch.no_grad():
    x, y = np.meshgrid(np.linspace(-w, w, density, dtype=np.float32), np.linspace(-h, h, density, dtype=np.float32))
    points = np.concatenate((np.expand_dims(x.flatten(), axis=1), np.expand_dims(y.flatten(), axis=1)), axis=1)
    points = torch.from_numpy(points).to(config.device, dtype=config.data.dtype)
    t = torch.zeros(points.shape[0], device=points.device)
    cond = (config.loss.std_min * (config.loss.std_max / config.loss.std_min) ** t)

    if config.model.type == 'energy_model':
        torch.set_grad_enabled(True)
        points_vf = score_fn.score_inference(points, cond).cpu().numpy()
        torch.set_grad_enabled(False)
    else:
        points_vf = score_fn(points, cond).cpu().numpy()

    fig = figure(figsize=(w/2, h/2), dpi=300)
    plt.quiver(x, y, points_vf[:,0], points_vf[:,1])
    plt.xticks([])
    plt.yticks([])
    plt.xlim((-w, w))
    plt.ylim((-h, h))
    plt.axis('off')
    plt.savefig(dir_file)
    plt.close(fig)

def plot_data_points(config, samples, dir_file):
  logging.info("Plotting Sampled Points...")
  w = config.sampling.width
  h = config.sampling.height
  fig = figure(figsize=(w/2, h/2), dpi=300)
  plt.xlim((-w, w))
  plt.ylim((-h, h))
  label = torch.zeros(samples.shape[0], dtype=torch.long)
  plot_data = np.vstack((samples.T, label)).T
  df = pd.DataFrame(data=plot_data, columns=("x", "y", "label"))
  sn.scatterplot(data=df, x="x", y="y", hue="label", alpha=0.8)
  plt.xticks([])
  plt.yticks([])
  plt.legend('')
  plt.savefig(dir_file)
  plt.close(fig)