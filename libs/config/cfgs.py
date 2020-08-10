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

# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
from enum import Enum

# ------------------------------------------------
# VERSION = 'FPN_Res101_20181201'
VERSION = 'FPN_Res101_20200728'
NET_NAME = 'resnet_v1_101'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "4"


SUMMARY_PATH = ROOT_PATH + '/outputs/summary'
INFERENCE_SAVE_PATH = ROOT_PATH + '/outputs/inference_results'
TEST_SAVE_PATH = ROOT_PATH + '/outputs/test_results'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/outputs/inference_image'


# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pascal'  # 'ship', 'spacenet', 'pascal', 'coco'
NUM_READER = 4
NUM_THREDS = 32
NUM_SPLIT_DATA = {
    'train': 17125,
    'val': 0,
}
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
NUM_CLASS = 20
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.
BBOX_CROP_OVERLAP = 0.5  # Minimum overlap to keep a bbox after cropping.
TRAIN_SIZE = (300, 300)
EVAL_SIZE = (300, 300)
# Some training pre-processing parameters.

DATA_FORMAT = ""

# Resizing strategies.
class Resize(Enum):
    NONE = 0 # Nothing!
    CENTRAL_CROP = 1 # Crop (and pad if necessary).
    PAD_AND_RESIZE = 2 # Pad, and resize to output shape.
    WARP_RESIZE = 3  # Warp resize.


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
