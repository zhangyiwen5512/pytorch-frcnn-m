# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
try:
  from model.config import cfg
  from roi_data_layer.minibatch import get_minibatch
except:
  from lib.model.config import cfg
  from lib.roi_data_layer.minibatch import get_minibatch
import numpy as np
import time

class RoIDataLayer(object):
  """Fast R-CNN data layer used for training."""

  def __init__(self, roidb, num_classes, random=False):
    """Set the roidb to be used by this layer during training."""
    self._roidb = roidb
    self._num_classes = num_classes
    # Also set a random flag
    self._random = random
    self._shuffle_roidb_inds()

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    # If the random flag is set, 
    # then the database is shuffled according to system time
    # Useful for the validation set
    if self._random:
      st0 = np.random.get_state()
      millis = int(round(time.time() * 1000)) % 4294967295
      np.random.seed(millis)
    
    if cfg.TRAIN.ASPECT_GROUPING:
      widths = np.array([r['width'] for r in self._roidb])
      heights = np.array([r['height'] for r in self._roidb])
      horz = (widths >= heights)
      vert = np.logical_not(horz)
      horz_inds = np.where(horz)[0]
      vert_inds = np.where(vert)[0]
      inds = np.hstack((
          np.random.permutation(horz_inds),
          np.random.permutation(vert_inds)))
      inds = np.reshape(inds, (-1, 2))
      row_perm = np.random.permutation(np.arange(inds.shape[0]))
      inds = np.reshape(inds[row_perm, :], (-1,))
      self._perm = inds
    else:
      self._perm = np.random.permutation(np.arange(len(self._roidb)))
    # Restore the random state
    if self._random:
      np.random.set_state(st0)
      
    self._cur = 0

  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
      self._shuffle_roidb_inds()

    # db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    if cfg.MIX_TRAINING:
      db_inds = self._perm[self._cur:self._cur + 2]
    else:
      db_inds = self._perm[self._cur:self._cur + 1]

    self._cur += cfg.TRAIN.IMS_PER_BATCH

    return db_inds

  def _get_next_minibatch(self):
    """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    db_inds = self._get_next_minibatch_inds()
    if cfg.MIX_TEST:
      print("TEST: db_inds: {}|".format(db_inds))
    minibatch_db = [self._roidb[i] for i in db_inds]
    return get_minibatch(minibatch_db, self._num_classes)
      
  def forward(self):
    """Get blobs and copy them into this layer's top blob vector."""
    blobs = self._get_next_minibatch()
    return blobs

if __name__ == "__main__":
  # one artificial image
  # (375, 500, 3)
  blob = {'data11':np.ones((3,4,3))}
  blob['image'] = 'data/000003.jpg'
  blob['boxes'] = np.array([[100,100,200,250],
                   [100, 100, 150, 300]])
  blob['gt_classes'] = np.array([1,2])
  blob['flipped'] = False

  blob2 = {'data11': np.ones((3, 4, 3))}
  blob2['image'] = 'data/000001.jpg'
  blob2['boxes'] = np.array([[10, 10, 20, 25],
                            [10, 10, 15, 30]])
  blob2['gt_classes'] = np.array([0, 1])
  blob2['flipped'] = False
  roidb = [blob, blob2]

  cfg.MIX_TRAINING = True

  datalayer = RoIDataLayer(roidb=roidb, num_classes=21)
  batch = datalayer.forward()
  print(batch)
  print(batch['data'].shape)
  # print(np.ones((2,3)))