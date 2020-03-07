#coding=utf-8
#!/usr/bin/python
import vot
from vot import Rectangle
#'''
import sys
sys.path.append("/home/guo/zpy/Mypysot")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from os.path import realpath, dirname, join
del os.environ['MKL_NUM_THREADS']

import time
import cv2
import torch
import numpy as np

from mypysot.tracker.TSDMTrack import TSDMTracker
from mypysot.tools.bbox import get_axis_aligned_bbox

# start to track
handle = vot.VOT("rectangle",'rgbd')
region = handle.region()
image_file1, image_file2 = handle.frame()
if not image_file1:
    sys.exit(0)

image_rgb = cv2.imread(image_file1)
image_depth = cv2.imread(image_file2, -1)
SiamRes_dir = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/modelMob.pth' #modelRes
SiamMask_dir = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/Mob20.pth' #Res20.pth'
Dr_dir = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/High-Low-two.pth.tar'
tracker = TSDMTracker(SiamRes_dir, SiamMask_dir, Dr_dir, image_rgb, image_depth, region)

#track
while True:
    image_file1, image_file2 = handle.frame()
    if not image_file1:
        break
    image_rgb = cv2.imread(image_file1)
    image_depth = cv2.imread(image_file2, -1)
    state = tracker.track(image_rgb, image_depth)
    region_Dr, score = state['region_Siam'], state['score'] #state['region_Dr']
    if score > 0.3:
        score = 1
    else:
        score = 0

    handle.report(Rectangle(region_Dr[0],region_Dr[1],region_Dr[2],region_Dr[3]), score)
'''
handle = vot.VOT("rectangle",'rgbd')
region = handle.region()
image_file1, image_file2 = handle.frame()
while True:
    handle.report(Rectangle(2,2,1,1), 1)
'''


