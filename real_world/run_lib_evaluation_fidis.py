import numpy as np
import os
import glob
import tensorflow as tf
import tensorflow_gan as tfgan
from ml_collections.config_flags import config_flags
from absl import flags
from absl import app
import logging

import datasets
import evaluation
import gc

from prdc import compute_prdc

FLAGS = flags.FLAGS

def create_encoded_files(config, workdir, encode_dataset=False):
  '''
  This function encodes the generated samples as well as the samples in `stat.npz' into
  the latents with a pretrained Inception model, and store them in a class-wise order.
  '''
  # Encodes the samples from the dataset.
  dir_name = "stat" + ("_ode" if FLAGS.ode else "")
  stat_dir = os.path.join(workdir, dir_name)
  tf.io.gfile.makedirs(stat_dir)
  dir_name = "eval_sampling"+ ("_ode" if FLAGS.ode else "")
  sampling_dir = os.path.join(workdir, dir_name)

  # Construct the inception model.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  if encode_dataset:
    # Convert dataset to stat file
    config.training.batch_size = config.eval.num_samples
    config.eval.batch_size = config.eval.num_samples
    _, training_ds, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)
    iter_ds = iter(training_ds)
    data = next(iter_ds)
    batch_all = data['image']._numpy()
    # Save the results in `pools` and `logits`
    pools = None
    logits = None
    # If the number of samples for each class are large, then it is split into mini-batches.
    num_encoding_rounds = config.eval.encoding_rounds
    batch_size = config.eval.num_samples // num_encoding_rounds
    for r in range(num_encoding_rounds):
      start = r*batch_size
      end = (r+1)*batch_size if r != num_encoding_rounds-1 else config.eval.num_samples
      batch = np.clip((batch_all[start:end, :, :, :]) * 255.0, 0, 255).astype(np.uint8)
      gc.collect()
      latent = evaluation.run_inception_distributed(batch, inception_model, inceptionv3=inceptionv3)
      gc.collect()
      pools = np.concatenate((pools, latent['pool_3']), axis=0) if pools is not None else latent['pool_3']
      logits = np.concatenate((logits, latent['logits']), axis=0) if logits is not None else latent['logits']

    # Save as .npz files
    np.savez(os.path.join(stat_dir, 'dataset_stat.npz'), pool_3=pools, logit=logits)
    logging.info("Finish encoding the samples from the dataset.")
    logging.info("="*10)
  
  # Encodes the generated samples.
  division_size = config.eval.num_samples // config.eval.num_divisions
  pools = None
  logits = None
  for d in range(config.eval.num_divisions):
    logging.info("Division: %d / %d" % (d+1, config.eval.num_divisions))
    logging.info("Sample Index from %d to %d." % (d*division_size, (d+1)*division_size))
    sample_di_dir = os.path.join(sampling_dir, str(d))
    for file in glob.glob(os.path.join(sample_di_dir, '*.npz')):
      logging.info("File Name: %s" % (file.split('/')[-1].replace(".npz", "")))
      sample = np.load(file)['samples']
      for minbatch in range(division_size // config.eval.batch_size):
        logging.info("Batch idx: %d" % (minbatch))
        gc.collect()
        latent = evaluation.run_inception_distributed(sample[minbatch*config.eval.batch_size:(minbatch+1)*config.eval.batch_size,:,:,:], inception_model, inceptionv3=inceptionv3)
        gc.collect()
        pools = np.concatenate((pools, latent['pool_3']), axis=0) if pools is not None else latent['pool_3']
        logits = np.concatenate((logits, latent['logits']), axis=0) if logits is not None else latent['logits']
  np.savez(os.path.join(stat_dir, 'generated_stat.npz'), pool_3=pools, logit=logits)
  logging.info("Finish encoding the generated samples.")

def read_nfe(config, workdir):
  dir_name = "eval_sampling"+ ("_ode" if FLAGS.ode else ("") )
  sampling_dir = os.path.join(workdir, dir_name)
  division_size = config.eval.num_samples // config.eval.num_divisions
  nfe_list = []
  for d in range(config.eval.num_divisions):
    logging.info("Division: %d / %d" % (d+1, config.eval.num_divisions))
    logging.info("Sample Index from %d to %d." % (d*division_size, (d+1)*division_size))
    sample_di_dir = os.path.join(sampling_dir, str(d))
    for file in glob.glob(os.path.join(sample_di_dir, '*.npz')):
      logging.info("File Name: %s" % (file.split('/')[-1].replace(".npz", "")))
      nfe = np.load(file)['nfe']
      nfe_list.append(nfe)
  logging.info("AVG NFE: %d" % (sum(nfe_list) / len(nfe_list)))

def evaluate_fidis(config, workdir, encode_dataset=False):
  '''
  This function reads the latent files and the `stat.npz' file, and calculates the FID / IS metrics.
  '''
  dir_name = "stat" + ("_ode" if FLAGS.ode else "")
  stat_dir = os.path.join(workdir, dir_name)
  # Load stats files
  if config.data.dataset == "CIFAR10":
    dataset_stat = np.load(os.path.join(stat_dir, 'dataset_stat.npz')) if encode_dataset else np.load("results/cifar10_stats.npz")
  elif config.data.dataset == "imagenet":
    dataset_stat = np.load(os.path.join(stat_dir, 'dataset_stat.npz')) if encode_dataset else np.load("results/imagenet_stats.npz")
  elif config.data.dataset == "CIFAR100":
    dataset_stat = np.load(os.path.join(stat_dir, 'dataset_stat.npz')) if encode_dataset else np.load("results/cifar100_stats.npz")
  elif config.data.dataset == "SVHN":
    dataset_stat = np.load(os.path.join(stat_dir, 'dataset_stat.npz')) if encode_dataset else np.load("results/svhn_stats.npz")
  else:
    raise ValueError("Dataset {} not recognized.".format(config.data.dataset))
  generated_stat = np.load(os.path.join(stat_dir, 'generated_stat.npz'))
  dataset_pool = dataset_stat['pool_3']
  generated_pool = generated_stat['pool_3'][:dataset_pool.shape[0],:]
  generated_logit = generated_stat['logit'][:dataset_pool.shape[0],:]
  # Compute the FID / IS metrics.
  fid = tfgan.eval.frechet_classifier_distance_from_activations(dataset_pool, generated_pool)
  inception_score = tfgan.eval.classifier_score_from_logits(generated_logit)
  logging.info("FID: %.2f || IS: %.2f" % (fid.numpy(), inception_score.numpy()))

def evaluate_prdc(config, workdir, encode_dataset=False):
  '''
  This function reads the latent files and the `stat.npz' file, and calculates the FID / IS metrics.
  '''
  nearest_k = config.eval.nearest_k
  dir_name = "stat" + ("_ode" if FLAGS.ode else "")
  stat_dir = os.path.join(workdir, dir_name)
  # Load stats files
  if config.data.dataset == "CIFAR10":
    dataset_stat = np.load(os.path.join(stat_dir, 'dataset_stat.npz')) if encode_dataset else np.load("results/stats/cifar10/stats.npz")
  elif config.data.dataset == "imagenet":
    dataset_stat = np.load(os.path.join(stat_dir, 'dataset_stat.npz')) if encode_dataset else np.load("results/stats/imagenet/stats.npz")
  elif config.data.dataset == "CIFAR100":
    dataset_stat = np.load(os.path.join(stat_dir, 'dataset_stat.npz')) if encode_dataset else np.load("results/stats/cifar100/stats.npz")
  elif config.data.dataset == "SVHN":
    dataset_stat = np.load(os.path.join(stat_dir, 'dataset_stat.npz')) if encode_dataset else np.load("results/stats/svhn/stats.npz")
  else:
    raise ValueError("Dataset {} not recognized.".format(config.data.dataset))
  generated_stat = np.load(os.path.join(stat_dir, 'generated_stat.npz'))
  dataset_pool = dataset_stat['pool_3']
  generated_pool = generated_stat['pool_3'][:dataset_pool.shape[0],:]
  # Compute FID / IS metrics.
  metrics = compute_prdc(real_features=dataset_pool, fake_features=generated_pool, nearest_k=nearest_k)
  logging.info("Precision: %2.2f || Recall:   %2.2f" % (metrics['precision'], metrics['recall']))
  logging.info("Density:   %2.2f || Coverage: %2.2f" % (metrics['density'], metrics['coverage']))

config_flags.DEFINE_config_file("config", None, "configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_boolean("encode", False, "Set as True to create encoded files.")
flags.DEFINE_boolean("encode_dataset", False, "Set as True to create encoded the images from the dataset.")
flags.DEFINE_boolean("fidis", False, "Set as True to calculate FID / IS.")
flags.DEFINE_boolean("prdc", False, "Set as True to calculate P / R / D / C.")
flags.DEFINE_boolean("ode", False, "Set as True to use ode solver.")
flags.DEFINE_boolean("nfe", False, "Set as True to display nfe.")
flags.DEFINE_integer("bs", 2500, "Batch size.")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
  config = FLAGS.config
  workdir = os.path.join('results', FLAGS.workdir)
  tf.io.gfile.makedirs(workdir)
  config.eval.batch_size = FLAGS.bs
  if FLAGS.encode:
    create_encoded_files(config, workdir, FLAGS.encode_dataset)
  if FLAGS.fidis:
    evaluate_fidis(config, workdir, FLAGS.encode_dataset)
  if FLAGS.prdc:
    evaluate_prdc(config, workdir, FLAGS.encode_dataset)
  if FLAGS.nfe:
    read_nfe(config, workdir)

if __name__ == "__main__":
  app.run(main)