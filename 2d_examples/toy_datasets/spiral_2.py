"""six center dataset."""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import numpy as np

_SIZE = 2
_NUM_POINTS = 50000
_DESCRIPTION = """
The 2d toy experiment dataset.
"""
_CITATION = """\
Nope.
"""

class spiral_2(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for inter_twinning_moon dataset."""

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
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'position': tfds.features.Tensor(shape=(_SIZE,), dtype=tf.float64),
            'label': tfds.features.ClassLabel(num_classes=1),
        }),
        supervised_keys=('position','label'),  # Set to `None` to disable
        homepage='None',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(split='all')),
    ]

  def _generate_examples(self, split):
    """Yields examples."""
    r = 2.3 / 3.0
    n = np.sqrt(np.random.rand(_NUM_POINTS // 2)) * 540 * (2 * np.pi) / 360
    d1x = - np.cos(n) * n + np.random.rand(_NUM_POINTS // 2) * 0.5
    d1y =   np.sin(n) * n + np.random.rand(_NUM_POINTS // 2) * 0.5
    positions = np.concatenate([np.stack([ d1x,  d1y], axis=1), np.stack([-d1x, -d1y], axis=1)], axis=0) * r

    labels = np.zeros(positions.shape[0], dtype=np.uint8)
    data = list(zip(positions, labels))
    for index, (position, label) in enumerate(data):
      record = {"position": position, "label": label}
      yield index, record