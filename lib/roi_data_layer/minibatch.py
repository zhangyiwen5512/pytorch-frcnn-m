# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
try:
  from model.config import cfg, tmp_lam
  from utils.blob import prep_im_for_blob, im_list_to_blob
except:
  from lib.model.config import cfg, tmp_lam
  from lib.utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  if cfg.MIX_TRAINING:
    im_blob, im_scales, trans_scales = _get_mix_image_blob(roidb, random_scale_inds)
    blobs = {'data': im_blob}
    assert len(im_scales) == 2, "MIX-TRAINING ERROR! Single batch only"

    if cfg.TRAIN.USE_ALL_GT:
      gt_inds  = np.where(roidb[0]['gt_classes'] != 0)[0]
      gt_inds2 = np.where(roidb[1]['gt_classes'] != 0)[0]
      if cfg.MIX_TEST:
        print("TEST: gt_inds {} 2 {}".format(gt_inds, gt_inds2))
    else:
      gt_inds  = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
      gt_inds2 = np.where(roidb[1]['gt_classes'] != 0 & np.all(roidb[1]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes          = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4]  = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4]    = roidb[0]['gt_classes'][gt_inds]

    gt_boxes2         = np.empty((len(gt_inds2), 5), dtype=np.float32)
    gt_boxes2[:, 0:4] = roidb[1]['boxes'][gt_inds2, :] * im_scales[1]
    gt_boxes2[:, 0] *= trans_scales[1]
    gt_boxes2[:, 1] *= trans_scales[0]
    gt_boxes2[:, 2] *= trans_scales[1]
    gt_boxes2[:, 3] *= trans_scales[0]
    gt_boxes2[:, 4]   = roidb[1]['gt_classes'][gt_inds2]

    blobs['gt_boxes'] = gt_boxes
    blobs['gt_boxes2'] = gt_boxes2
    blobs['im_info'] = np.array(
      [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
      dtype=np.float32)

    return blobs

  else:
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    # assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
      # Include all ground truth boxes
      gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
      if cfg.MIX_TEST:
        print("TEST: gt_inds {} ".format(gt_inds))
    else:
      # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
      gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    if cfg.MIX_TEST:
      print("TEST: gt_boxes {} ".format(gt_boxes))
      print(roidb[0]['gt_classes'][gt_inds])
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
      [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
      dtype=np.float32)

    return blobs


def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

def _get_mix_image_blob(roidb, scale_inds):
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # add two image
  im1 = processed_ims[0]
  im2_tmp = processed_ims[1]
  s1, s2 = np.array(im1.shape, dtype=np.float32), np.array(im2_tmp.shape, dtype=np.float32)
  trans_scales = s1 / s2
  im2 = cv2.resize(im2_tmp, None, None, fx=trans_scales[1], fy=trans_scales[0],
             interpolation=cv2.INTER_LINEAR)
  assert im1.shape == im2.shape, " {}  ?  {}  tr scales {},  s1,s2   {} ? {}".format(im1.shape, im2.shape, trans_scales, s1, s2)

  lam = tmp_lam
  im = lam * im1 + (1-lam) * im2

  processed_ims = [im]
  # im_scales

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales, trans_scales