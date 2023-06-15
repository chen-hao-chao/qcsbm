"""imagenet dataset."""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import numpy as np
import pickle
import os

class imagenet(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ImageNet32x32 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=(32, 32, 3), dtype=tf.float32),
            'label': tfds.features.ClassLabel(num_classes=1000),
        }),
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "split": "train",
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "split": "val",
            }),
    ]

  def _generate_examples(self, split):
    """Yields examples."""
    data_folder = "imagenet"
    images = None
    labels = None
    if split == "train":
        for idx in range(10):
            data_file = os.path.join(data_folder, 'train_data_batch_')
            d = unpickle(data_file + str(idx+1))
            image, label = process(d)
            labels = np.concatenate((labels, label), axis=0) if labels is not None else label
            images = np.concatenate((images, image), axis=0) if images is not None else image
    elif split == "val":
        data_file = os.path.join(data_folder, 'val_data')
        d = unpickle(data_file)
        images, labels = process(d)
    else:
        raise ValueError("Set not recognized.")
    
    data = list(zip(images, labels))
    for index, (image, label) in enumerate(data):
        record = {"image": image, "label": label}
        yield index, record

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def process(d, img_size=32):
    x = d['data']
    y = d['labels']
    x = x/np.float32(255)
    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    y = np.asarray(y)
    img_size2 = img_size * img_size
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    
    return x, y