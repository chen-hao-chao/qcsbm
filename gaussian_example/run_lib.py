import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import tensorflow as tf
import torch
from torch.autograd import Variable

def mixture_gaussian(points, num=10, r=3, sigma=1):
  l = [2*np.pi*(i/num) for i in range(num)]
  prob = torch.zeros(points.shape[0]).to(points.device)
  for li in range(len(l)):
    x0 = r * np.cos(l[li])*torch.ones(points.shape[0]).to(points.device)
    x1 = r * np.sin(l[li])*torch.ones(points.shape[0]).to(points.device)
    prob += (1 / num) * torch.exp(  - ( ( (points[:, 0]-x0)**2 + (points[:, 1]-x1)**2 )  / (2*(sigma**2)) )   )  /  ( 2*np.pi*(sigma**2) ) 
  return prob

def curl_score(points, sigma, epsilon, eps=1e-8, fn='usbm'):
  # True gradients
  grad_logp = - points / (sigma**2)
  # True probability
  squared_norm = points[:, 0]**2 + points[:, 1]**2
  p = torch.exp(  - ( squared_norm / (2*(sigma**2)) ) )  /  ( 2*np.pi*(sigma**2) )
  # Perturbation vector
  if epsilon != 0.0:
    mu = mixture_gaussian(points)
    if fn == "usbm":
      u1 =  - points[:, 1] * torch.sqrt( 2*(epsilon+1e-6)*mu / (p*(squared_norm + (epsilon+1e-6))+eps) )
      u2 =  points[:, 0] * torch.sqrt( 2*(epsilon+1e-6)*mu / (p*(squared_norm + (epsilon+1e-6))+eps) )
    else: # fn == "csbm"
      u1 = points[:, 0] * torch.sqrt( 2*(epsilon+1e-6)*mu / (p*(squared_norm + (epsilon+1e-6))+eps) )
      u2 = points[:, 1] * torch.sqrt( 2*(epsilon+1e-6)*mu / (p*(squared_norm + (epsilon+1e-6))+eps) )
    u = torch.stack((u1, u2), -1)
    grad_logp = u + grad_logp
  return grad_logp, p

def plot_traj(traj, sigma, epsilon, points, x, y, w, h, num_traj=10, steps=100, traj_dir="traj_dir", fn="usbm"):
  # Plotting settings
  color_traj = ['cornflowerblue', 'navy', 'dodgerblue', 'deepskyblue', 'powderblue', 
                  'turquoise', 'aquamarine', 'lightseagreen', 'royalblue', 'slateblue']
  figure(figsize=(5, 5), dpi=300)
  plt.xlim((-w, w))
  plt.ylim((-h, h))
  plt.legend('')

  # Plot vector field.
  with torch.no_grad():
    grad_logp, _ = curl_score(points, sigma=sigma, epsilon=epsilon, fn=fn)
  grad_logp = grad_logp.cpu().numpy()
  plt.quiver(x, y, grad_logp[:,0], grad_logp[:,1])
  
  # Plot trajectory.
  for i in [int(steps)-1]:
    for j in range(num_traj):
      trajectory = traj[:, j : (i+1)*num_traj+j : num_traj]
      plt.plot(trajectory[0, :], trajectory[1, :], 'x-', color=color_traj[j%len(color_traj)], linewidth=1.5, markersize=5)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(traj_dir, str(i)+".png"))
    plt.savefig(os.path.join(traj_dir, str(i)+".eps"))

def sampling_fn(data, sigma, epsilon, stochastic=True, device='cuda', steps=1.0e3, step_size=1.0e-3, div=100, eps=1e-16, metropolis=False, dtype=torch.float32, fn="usbm"):
  logp = list()
  traj = data.T
  flag = True
  min_step = steps
  step_size = 0.0125 if epsilon>0 else 0.01
  with torch.no_grad():
    x = torch.from_numpy(data).to(device, dtype=dtype)
    shape = x.shape
    for i in range(int(steps)):
      cur_step_size = step_size / np.exp(i/1000)
      dx, p = curl_score(x, sigma, epsilon, fn=fn)
      # Perform one-step Langevin dynamics.
      xp = x + dx*cur_step_size + torch.randn_like(x)*np.sqrt(cur_step_size*2) if stochastic else x + dx*cur_step_size
      # Adjust the direction using the Metropolis method.
      if metropolis:
        diff = (xp - x).unsqueeze(0).expand(div, shape[0], shape[1])
        part = torch.range(0.0, 1.0-1.0/div, 1.0/div).unsqueeze(1).unsqueeze(1).expand(div, shape[0], shape[1]).to(x.device)
        delta = diff * part
        xp_mid = (x.unsqueeze(0).expand(div, shape[0], shape[1]) + delta).contiguous().view(div*shape[0], shape[1])
        score_mid, _ = curl_score(xp_mid, sigma, epsilon)
        energy_diff = torch.sum(score_mid.contiguous().view(div, shape[0], shape[1]) * diff, dim=(0,2)) * (1/div)
        alpha = torch.exp(energy_diff).unsqueeze(1).expand(shape[0], shape[1])
        u = torch.rand(alpha.shape).to(x.device)
        x = torch.where(u <= alpha, xp, x)
      else:
        x = xp

      nll = np.mean( np.clip(-np.log(p.cpu().numpy() + eps), a_min=0, a_max=np.inf))
      logp.append(nll)
      traj = np.concatenate((traj, x.cpu().numpy().T), axis=1)
      if nll < 0.001 and flag:
        flag = False
        min_step = i

  logging.info("Required number of steps with error=%.2e is: %d" % (epsilon, min_step))
  return traj

def plot_curl_traj(config, workdir):
  # Initialize variables.
  w = config.sampling.width
  h = config.sampling.height
  d = config.sampling.density
  sigma = config.sampling.std
  num_traj = config.sampling.num_traj
  steps = config.sampling.steps
  init_position = config.sampling.init_position
  init_variance = config.sampling.init_variance
  fn = config.sampling.type
  shape = (num_traj, 2)
  epsilon_list = [0.0e-3, 0.5e-3, 0.75e-3, 1.0e-3, 1.5e-3, 1.75e-3, 2.25e-3]
  dtype = np.double
  dtype_torch = torch.double
  
  # Construct the coordinates for evaluation.
  x, y = np.meshgrid(np.linspace(-w, w, d, dtype=dtype), np.linspace(-h, h, d, dtype=dtype))
  points = torch.tensor(np.concatenate((np.expand_dims(x.flatten(), axis=1), np.expand_dims(y.flatten(), axis=1)), axis=1)).to(config.device)
  init = np.random.normal(init_position, init_variance, shape[0]*shape[1]).reshape(shape)
  
  for epsilon in epsilon_list:
    # Initiate a new folder.
    traj_dir = os.path.join(workdir, "traj_eps="+str(epsilon))
    tf.io.gfile.makedirs(traj_dir)

    # Plot trajectories  
    traj = sampling_fn(init, sigma, epsilon, steps=steps, stochastic=False, dtype=dtype_torch, fn=fn)
    plot_traj(traj, sigma, epsilon, points, x, y, w, h, num_traj, steps=steps, traj_dir=traj_dir, fn=fn)

    # Calculate symmetry distance
    points = Variable(points, requires_grad=True)
    score, prob = curl_score(points, sigma, epsilon, fn=fn)
    grad0, = torch.autograd.grad(torch.sum(score[torch.arange(score.shape[0]), 0]), points, retain_graph=True)
    grad1, = torch.autograd.grad(torch.sum(score[torch.arange(score.shape[0]), 1]), points)
    traceJJt = torch.sum(torch.square(grad0), dim=1) + torch.sum(torch.square(grad1), dim=1)
    traceJJ = torch.square(grad0[:,0]) + torch.square(grad1[:,1]) + 2*grad0[:,1]*grad1[:,0]
    squared_asym_norm = 0.5 * (traceJJt - traceJJ)
    nasym = (squared_asym_norm / traceJJt).cpu().numpy()
    asym = (traceJJt - traceJJ).cpu().numpy()
    prob = prob.detach().cpu().numpy()
    logging.info("Sampling process with error=%.2e have been accomplished." % (epsilon))
    logging.info("NAsym: %.2e || Asym: %.2e." % (np.sum(nasym*prob), np.sum(asym*prob)))
    logging.info("="*10)

def run(config, workdir):
  plot_curl_traj(config, workdir)