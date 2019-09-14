#!/usr/bin/env python
# encoding: utf-8
# Author: yongyuan.name

import os
import cv2
import multiprocessing
from multiprocessing import Process, freeze_support, Pool

import torch
import sosnet_model
import tfeat_utils
import numpy as np
torch.no_grad()

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
            for i in range(wanted_parts) ]

def gpu_task(img_names, db_dir, save_dir):
    sosnet32 = sosnet_model.SOSNet32x32()
    net_name = 'notredame'
    sosnet32.load_state_dict(torch.load(os.path.join('sosnet-weights',"sosnet-32x32-"+net_name+".pth")))
    sosnet32.cuda().eval()

    local_detector = cv2.xfeatures2d.SIFT_create()

    for i, line in enumerate(img_names):
        img_path = os.path.join(db_dir, line)
        print img_path
        img = cv2.imread(img_path, 1)
        height, width = img.shape[:2]
        img_resize = cv2.resize(img, (int(0.5*width), int(0.5*height)))
        kpt = local_detector.detect(img, None)
        desc = tfeat_utils.describe_opencv(sosnet32, \
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), kpt, \
                patch_size = 32, mag_factor = 7, use_gpu = True)
        with open(os.path.join(save_dir, line.split('.jpg')[0] + '.sosnet.sift'), 'w') as f:
            if desc is None:
                f.write(str(128) + '\n')
                f.write(str(0) + '\n')
                f.close()
                print "Null: %s" % line
                continue
            if len(desc) > 0:
                f.write(str(128) + '\n')
                f.write(str(len(kpt)) + '\n')
                for j in range(len(desc)):
                    locs_str = '0 0 0 0 0 '
                    descs_str = " ".join([str(float(value)) for value in desc[j]])
                    all_strs = locs_str + descs_str
                    f.write(all_strs + '\n')
                f.close()
            print "%d(%d), %s, desc: %d" %(i+1, len(img_names), line, len(desc))

if __name__ == '__main__':

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()

    parts = 1
    txt_path = './cbir_public_datasets/oxford/oxford.txt'
    db_dir = './cbir_public_datasets/oxford/jpg'
    save_dir = './cbir_public_datasets/oxford/sosnet'

    with open(txt_path, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    blocks = split_list(content, wanted_parts = parts)
    gpu_task(blocks[0], db_dir, save_dir)
    #for i in xrange(0, parts):
    #    pool.apply_async(gpu_task, args=(blocks[i], db_dir, save_dir,))
    #pool.close()
    #pool.join()
