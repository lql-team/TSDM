import time
import cv2
import torch
import numpy as np
import argparse
import sys
import os

sys.path.append("../")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


from mypysot.tracker.TSDMTrack import TSDMTracker
from mypysot.tools.bbox import get_axis_aligned_bbox

parser = argparse.ArgumentParser(description='TSDM is running')
parser.add_argument('--data_dir', default='data_and_result/test_data')
parser.add_argument('--sequence', default=1, type=int)

name = ['start/']# not be used
gt_bbox = [[1,1,1,1]]
name.append('teapot/') #1
gt_bbox.append([318.67,133.0,119.67,97.0 ]) 
name.append('XMG_outside/') #2
gt_bbox.append([338.23,164.54,98.909,54.001])
name.append('box_darkroom_noocc_1/') #3
gt_bbox.append([219.2,173.88,143.6,95.4])

args = parser.parse_args()
name = name[args.sequence]
gt_bbox = gt_bbox[args.sequence]


#init
args.data_dir
image_file1 = args.data_dir + name + 'color/00000001.jpg'
image_file2 = args.data_dir + name + 'depth/00000001.png'
if not os.path.exists(image_file1):
    sys.exit(0)
image_rgb = cv2.imread(image_file1)
image_depth = cv2.imread(image_file2, -1)
SiamRes_dir = 'data_and_result/weight/modelRes.pth' # for not retraining core
SiamMask_dir = 'data_and_result/weight/Res20.pth' # for retraining core
Dr_dir = 'data_and_result/weight/High-Low-two.pth.tar' # for D-r, an information fusion network 
tracker = TSDMTracker(SiamRes_dir, SiamMask_dir, Dr_dir, image_rgb, image_depth, gt_bbox)

#track
sumTime = 0
for i in range(2,5000):
    #track of the first stage
    image_file1 = args.data_dir + name + 'color/'
    image_file1 = image_file1 + str(i).zfill(8) + '.jpg'
    image_file2 = args.data_dir + name + 'depth/'
    image_file2 = image_file2 + str(i).zfill(8) + '.png'
    image_rgb = cv2.imread(image_file1)
    image_depth = cv2.imread(image_file2, -1)
    if not os.path.exists(image_file1):
        print('Avg. speed:')
        print(i/(sumTime+0.01))
        break

    start = time.perf_counter()
    state = tracker.track(image_rgb, image_depth)
    region_Siam, region_nms, region_Dr = state['region_Siam'], state['region_nms'], state['region_Dr']

    eachTime = time.perf_counter() - start
    sumTime += eachTime
    print("Using time:",eachTime)
    
    save_path = '/home/guo/zpy/Mypysot/mypysot/data_and_result/result/TSDM_' + str(i).zfill(8) + '.jpg'
    img = state['Xm']
    region = region_Siam
    x1, y1 = int(region[0]), int(region[1])
    x2, y2 = int(region[0]+region[2]), int(region[1]+region[3])
    cv2.line(img,(x1,y1),(x2,y1),(255,0,0),1)
    cv2.line(img,(x1,y2),(x2,y2),(255,0,0),1)
    cv2.line(img,(x1,y1),(x1,y2),(255,0,0),1)
    cv2.line(img,(x2,y1),(x2,y2),(255,0,0),1)

    region = region_nms
    x1, y1 = int(region[0]), int(region[1])
    x2, y2 = int(region[0]+region[2]), int(region[1]+region[3])
    cv2.line(img,(x1,y1),(x2,y1),(0,255,255),1)
    cv2.line(img,(x1,y2),(x2,y2),(0,255,255),1)
    cv2.line(img,(x1,y1),(x1,y2),(0,255,255),1)
    cv2.line(img,(x2,y1),(x2,y2),(0,255,255),1)

    region = region_Dr
    x1, y1 = int(region[0]), int(region[1])
    x2, y2 = int(region[0]+region[2]), int(region[1]+region[3])
    cv2.line(img,(x1,y1),(x2,y1),(0,0,255),1)
    cv2.line(img,(x1,y2),(x2,y2),(0,0,255),1)
    cv2.line(img,(x1,y1),(x1,y2),(0,0,255),1)
    cv2.line(img,(x2,y1),(x2,y2),(0,0,255),1)

    text =  'target score: ' + str(round(state['score'], 3))
    img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(save_path, img)
    
    print(i)

'''
# Copyright (c) SenseTime. All Rights Reserved.
import sys
sys.path.append("/home/guo/zpy/Mypysot")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import time
import cv2
import torch
import numpy as np
import argparse

from mypysot.tracker.TSDMTrack import TSDMTracker
from mypysot.tools.bbox import get_axis_aligned_bbox


parser = argparse.ArgumentParser(description='TSDM is running')
parser.add_argument('--sequence', default=1, type=int)

name = ['start/']# not be used
gt_bbox = [[1,1,1,1]]
name.append('teapot/') #1
gt_bbox.append([318.67,133.0,119.67,97.0 ]) 
name.append('XMG_outside/') #2
gt_bbox.append([338.23,164.54,98.909,54.001])
name.append('box_darkroom_noocc_1/') #3
gt_bbox.append([219.2,173.88,143.6,95.4])
name.append('robot_human_corridor_noocc_1_A/') #4
gt_bbox.append([124.0,272.88,196.4,92.52])

name.append('box1_outside/') #5 2501
gt_bbox.append([442.18,207.03,40.242,127.75])
name.append('XMG_outside/') #6 1910
gt_bbox.append([338.23,164.54,98.909,54.001])
name.append('trophy_outside/') #7 2001
gt_bbox.append([445.58,189.82,53.819,77.921])
name.append('robot_human_corridor_noocc_1_B/') #8 1430
gt_bbox.append([79.0,69.3,132.0,200.1])
name.append('box_darkroom_noocc_1/') #9 1165
gt_bbox.append([219.2,173.88,143.6,95.4])





args = parser.parse_args()
name = name[args.sequence]
gt_bbox = gt_bbox[args.sequence]

def IOU(box1, box2):
    x1i = max(box1[0], box2[0])
    y1i = max(box1[1], box2[1])
    x2i = min(box1[2], box2[2])
    y2i = min(box1[3], box2[3])
    x1u = min(box1[0], box2[0])
    y1u = min(box1[1], box2[1])
    x2u = max(box1[2], box2[2])
    y2u = max(box1[3], box2[3])
    if x1i >= x2i or y1i >= y2i:
        return 0
    else:
        return (x2i-x1i)*(y2i-y1i)/(x2u-x1u)/(y2u-y1u)

#init
data_root = '/media/guo/DataUbuntu/vot-toolkit-master/sequences/'
image_file1 = data_root + name + 'color/00000001.jpg'
image_file2 = data_root + name + 'depth/00000001.png'
if not os.path.exists(image_file1):
    sys.exit(0)
image_rgb = cv2.imread(image_file1)
image_depth = cv2.imread(image_file2, -1)
SiamRes_dir = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/modelRes.pth' #modelRes.pth modelMob.pth
SiamMask_dir = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/modelRes.pth' # Res20.pth Mob20.pth
Dr_dir = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/High-Low-two.pth.tar'
tracker = TSDMTracker(SiamRes_dir, SiamMask_dir, Dr_dir, image_rgb, image_depth, gt_bbox)


region_gt = []
with open(data_root + name + 'groundtruth.txt', 'r') as f:
    lines = f.readlines()
    for j in range(0, len(lines)):
        region_gt.append([float(i) for i in lines[j].strip('\n').split(',')])
        region_gt[j] = [region_gt[j][0], region_gt[j][1], region_gt[j][0] + region_gt[j][2], region_gt[j][1] + region_gt[j][3]]
        if region_gt[j][0] != region_gt[j][0]:
            region_gt[j] = [-5, -4, -5, -4]

#track
sumTime = 0
total_iou = 0
total_noocc = 0
for i in range(2,5000):
    #track of the first stage
    image_file1 = data_root + name + 'color/'
    image_file1 = image_file1 + str(i).zfill(8) + '.jpg'
    image_file2 = data_root + name + 'depth/'
    image_file2 = image_file2 + str(i).zfill(8) + '.png'
    image_rgb = cv2.imread(image_file1)
    image_depth = cv2.imread(image_file2, -1)
    if not os.path.exists(image_file1):
        print('Avg. speed:')
        print(i/(sumTime+0.01))
        print(total_iou/total_noocc)
        break

    start = time.perf_counter()
    state = tracker.track(image_rgb, image_depth)
    region_Siam, region_nms, region_Dr = state['region_Siam'], state['region_nms'], state['region_Dr']

    eachTime = time.perf_counter() - start
    sumTime += eachTime
    print("Using time:",eachTime)
    
    save_path = '/home/guo/zpy/Mypysot/mypysot/data_and_result/result/result2/TSDM_' + str(i).zfill(8) + '.jpg'
    img = state['Xm'] #state['Xm'] image_rgb.copy()
    region = region_Siam
    x1, y1 = int(region[0]), int(region[1])
    x2, y2 = int(region[0]+region[2]), int(region[1]+region[3])
    cv2.line(img,(x1,y1),(x2,y1),(255,0,0),1)
    cv2.line(img,(x1,y2),(x2,y2),(255,0,0),1)
    cv2.line(img,(x1,y1),(x1,y2),(255,0,0),1)
    cv2.line(img,(x2,y1),(x2,y2),(255,0,0),1)

    region = region_nms
    x1, y1 = int(region[0]), int(region[1])
    x2, y2 = int(region[0]+region[2]), int(region[1]+region[3])
    cv2.line(img,(x1,y1),(x2,y1),(0,255,255),1)
    cv2.line(img,(x1,y2),(x2,y2),(0,255,255),1)
    cv2.line(img,(x1,y1),(x1,y2),(0,255,255),1)
    cv2.line(img,(x2,y1),(x2,y2),(0,255,255),1)

    region = region_Dr
    x1, y1 = int(region[0]), int(region[1])
    x2, y2 = int(region[0]+region[2]), int(region[1]+region[3])
    cv2.line(img,(x1,y1),(x2,y1),(0,0,255),1)
    cv2.line(img,(x1,y2),(x2,y2),(0,0,255),1)
    cv2.line(img,(x1,y1),(x1,y2),(0,0,255),1)
    cv2.line(img,(x2,y1),(x2,y2),(0,0,255),1)

    text =  'target score: ' + str(round(state['score'], 3))
    img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(save_path, img)
    
    iou = IOU([x1, y1, x2, y2], region_gt[i-1])
    print(iou)
    total_iou += iou
    if region_gt[i-1][0] != -5:
        total_noocc += 1

    print(i)
'''