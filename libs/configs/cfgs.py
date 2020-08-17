#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cfgs.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/8/4 下午3:22
# @ Software   : PyCharm
#-------------------------------------------------------
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
from enum import Enum

# ------------------------------------------------
# VERSION = 'FPN_Res101_20181201'
VERSION = 'SSD_300_VGG_20200812'
MODEL_NAME = 'ssd_300_vgg'
BASE_NETWORK_NAME = 'vgg_16'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "4"

SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100  # 'The frequency with which logs are print.')
SAVE_WEIGHTS_INTE = 10000

NUM_CLONES = 1  # Number of model clones to deploy
CLONE_ON_CPU = False # 'Use CPUs to deploy clones.'
GPU_MEMORY_FRACTION = 0.8  # 'GPU memory fraction to use.'


TFRECORD_DIR = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/pascal_tfrecord_ssd'
SUMMARY_PATH = ROOT_PATH + '/outputs/summary'
INFERENCE_SAVE_PATH = ROOT_PATH + '/outputs/inference_results'
TEST_SAVE_PATH = ROOT_PATH + '/outputs/test_results'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/outputs/inference_image'

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'outputs/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/outputs/evaluate_result'
# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pascal'  # 'ship', 'spacenet', 'pascal', 'coco'
DATA_FORMAT = 'NCHW'
PIXEL_MEAN = [123.68, 116.78, 103.94]
BATCH_SIZE = 16
NUM_READER = 4  # The number of parallel readers that read data from the dataset.
NUM_THREADS = 4  # 'The number of threads used to create the batches.'
NUM_SPLIT_DATA = {
    'train': 17125,
    'val': 0,
}
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.
BBOX_CROP_OVERLAP = 0.5  # Minimum overlap to keep a bbox after cropping.
TRAIN_SIZE = (300, 300)
EVAL_SIZE = (300, 300)
# Some training pre-processing parameters.
# Resizing strategies.
class Resize(Enum):
    NONE = 0 # Nothing!
    CENTRAL_CROP = 1 # Crop (and pad if necessary).
    PAD_AND_RESIZE = 2 # Pad, and resize to output shape.
    WARP_RESIZE = 3  # Warp resize.

#-----------------------------------network config------------------------------------
IMAGE_SHAPE = (300, 300)
NUM_CLASS = 20
NO_ANNOTATION_LABEL = 21
FEATURE_LAYER = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
FEATURE_SHAPE = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
ANCHOR_SIZE_BOUND = [0.15, 0.90]
# anchor_size_bounds=[0.20, 0.90],
ANCHOR_SIZE = [(21., 45.),
                (45., 99.),
                (99., 153.),
                (153., 207.),
                (207., 261.),
                (261., 315.)]
# anchor_sizes=[(30., 60.),
#               (60., 111.),
#               (111., 162.),
#               (162., 213.),
#               (213., 264.),
#               (264., 315.)],
ANCHOR_RATIO = [[2, .5],
                 [2, .5, 3, 1. / 3],
                 [2, .5, 3, 1. / 3],
                 [2, .5, 3, 1. / 3],
                 [2, .5],
                 [2, .5]]
ANCHOR_STEPS = [8, 16, 32, 64, 100, 300]
ANCHOR_OFFSETS = 0.5
NORMALIZATION = [20, -1, -1, -1, -1, -1]
PRIOR_SCALING = [0.1, 0.1, 0.2, 0.2]

#---------------------------ssd net flag-----------------------------
LOSS_ALPHA = 1  # Alpha para meter in the loss function.
NEGATIVE_RATIO = 3. # 'Negative ratio in the loss function.'
MATCH_THRESHOLD = 0.5 # Matching threshold in the loss function.

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
# use momentum optimizer
WEIGHT_DECAY = 0.0005 # The weight decay on the model weights.
MOMENTUM = 0.9  # The momentum for the MomentumOptimizer and RMSPropOptimizer.
LEARING_RATE_BASE = 0.001
WARM_UP_LEARING_RATE = 0.0001
END_LEARNING_RATE = 0.000001
DECAY_STEP = [12000, 16000]  # 50000, 70000
WARM_UP_STEP = 8000

EPSILON = 1e-5
MAX_ITERATION = 200000

LABELS_SMOOTH = 0.0  # The amount of label smoothing.
MOVING_AVERATE_DECAY = None  # The decay to use for the moving average.


# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
CHEACKPOINT_PATH = None  # The path to a checkpoint from which to fine-tune.
CHEACKPOINT_MODEL_SCOPE = None  # Model scope in the checkpoint. None if the same as the trained model.
CHEACKPOINT_EXCLUDE_SCOPES = None  #Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.
TRAINABLE_SCOPE = None  # Comma-separated list of scopes to filter the set of variables to train. By default, None would train all the variables.
IGNORE_MISSING_VARS = False  # When restoring a checkpoint would ignore missing variables.



VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

COCO_LABELS = {
    "bench":  (14, 'outdoor') ,
    "skateboard":  (37, 'sports') ,
    "toothbrush":  (80, 'indoor') ,
    "person":  (1, 'person') ,
    "donut":  (55, 'food') ,
    "none":  (0, 'background') ,
    "refrigerator":  (73, 'appliance') ,
    "horse":  (18, 'animal') ,
    "elephant":  (21, 'animal') ,
    "book":  (74, 'indoor') ,
    "car":  (3, 'vehicle') ,
    "keyboard":  (67, 'electronic') ,
    "cow":  (20, 'animal') ,
    "microwave":  (69, 'appliance') ,
    "traffic light":  (10, 'outdoor') ,
    "tie":  (28, 'accessory') ,
    "dining table":  (61, 'furniture') ,
    "toaster":  (71, 'appliance') ,
    "baseball glove":  (36, 'sports') ,
    "giraffe":  (24, 'animal') ,
    "cake":  (56, 'food') ,
    "handbag":  (27, 'accessory') ,
    "scissors":  (77, 'indoor') ,
    "bowl":  (46, 'kitchen') ,
    "couch":  (58, 'furniture') ,
    "chair":  (57, 'furniture') ,
    "boat":  (9, 'vehicle') ,
    "hair drier":  (79, 'indoor') ,
    "airplane":  (5, 'vehicle') ,
    "pizza":  (54, 'food') ,
    "backpack":  (25, 'accessory') ,
    "kite":  (34, 'sports') ,
    "sheep":  (19, 'animal') ,
    "umbrella":  (26, 'accessory') ,
    "stop sign":  (12, 'outdoor') ,
    "truck":  (8, 'vehicle') ,
    "skis":  (31, 'sports') ,
    "sandwich":  (49, 'food') ,
    "broccoli":  (51, 'food') ,
    "wine glass":  (41, 'kitchen') ,
    "surfboard":  (38, 'sports') ,
    "sports ball":  (33, 'sports') ,
    "cell phone":  (68, 'electronic') ,
    "dog":  (17, 'animal') ,
    "bed":  (60, 'furniture') ,
    "toilet":  (62, 'furniture') ,
    "fire hydrant":  (11, 'outdoor') ,
    "oven":  (70, 'appliance') ,
    "zebra":  (23, 'animal') ,
    "tv":  (63, 'electronic') ,
    "potted plant":  (59, 'furniture') ,
    "parking meter":  (13, 'outdoor') ,
    "spoon":  (45, 'kitchen') ,
    "bus":  (6, 'vehicle') ,
    "laptop":  (64, 'electronic') ,
    "cup":  (42, 'kitchen') ,
    "bird":  (15, 'animal') ,
    "sink":  (72, 'appliance') ,
    "remote":  (66, 'electronic') ,
    "bicycle":  (2, 'vehicle') ,
    "tennis racket":  (39, 'sports') ,
    "baseball bat":  (35, 'sports') ,
    "cat":  (16, 'animal') ,
    "fork":  (43, 'kitchen') ,
    "suitcase":  (29, 'accessory') ,
    "snowboard":  (32, 'sports') ,
    "clock":  (75, 'indoor') ,
    "apple":  (48, 'food') ,
    "mouse":  (65, 'electronic') ,
    "bottle":  (40, 'kitchen') ,
    "frisbee":  (30, 'sports') ,
    "carrot":  (52, 'food') ,
    "bear":  (22, 'animal') ,
    "hot dog":  (53, 'food') ,
    "teddy bear":  (78, 'indoor') ,
    "knife":  (44, 'kitchen') ,
    "train":  (7, 'vehicle') ,
    "vase":  (76, 'indoor') ,
    "banana":  (47, 'food') ,
    "motorcycle":  (4, 'vehicle') ,
    "orange":  (50, 'food')
  }


PASCAL_NAME_LABEL_MAP = {
    'back_ground': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}
