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

class gaussian_8(tfds.core.GeneratorBasedBuilder):
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
    labels = np.zeros((_NUM_POINTS, ))

    scale = 8
    sq2 = 1 / np.sqrt(2)
    centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
    centers = np.array([(scale * x, scale * y) for x,y in centers])
    positions = sq2 * (np.random.randn(_NUM_POINTS, 2)+centers[np.random.randint(0, len(centers), size=(_NUM_POINTS,))])

    data = list(zip(positions, labels))
    for index, (position, label) in enumerate(data):
      record = {"position": position, "label": label}
      yield index, record