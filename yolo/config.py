# -*- coding:utf-8 -*-
#
# Created by lee
#
# 2018-04-16

import os
#
# path and dataset parameter
#
DATA_PATH = 'data'
PASCAL_PATH = os.path.join(DATA_PATH, 'Pascal_voc')
OUTPUT_DIR = os.path.join(DATA_PATH, 'output')
WEIGHTS = 'YOLO_small.ckpt'

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448
CELL_SIZE = 7
BOXES_PER_CELL = 2

ALPHA = 0.1

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

GPU = ''

LEARNING_RATE = 0.0001
DECAY_STEPS = 20000
DECAY_RATE = 0.1
STAIRCASE = True

BATCH_SIZE = 32
MAX_STEP = 30000
SUMMARY_STEP = 10
SAVE_STEP = 50

THRESHOLD = 0.2
IOU_THRESHOLD = 0.5
