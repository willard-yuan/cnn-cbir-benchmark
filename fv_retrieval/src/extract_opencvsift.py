#!/usr/bin/env python
# encoding: utf-8
# Author: yongyuan.name

import os
import cv2
import multiprocessing
from multiprocessing import Process, freeze_support, Pool


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def cpu_task(img_names, db_dir, save_dir):
    sift = cv2.xfeatures2d.SIFT_create()
    for i, line in enumerate(img_names):
        img_path = os.path.join(db_dir, line)
        print img_path
        img = cv2.imread(img_path, 0)
        height, width = img.shape[:2]
        img_resize = cv2.resize(img, (int(0.5*width), int(0.5*height)))
        kp, des = sift.detectAndCompute(img_resize, None)
        with open(os.path.join(save_dir, line.split('.jpg')[0] + '.opencv.sift'), 'w') as f:
            if des is None:
                f.write(str(128) + '\n')
                f.write(str(0) + '\n')
                f.close()
                print "Null: %s" % line
                continue
            if len(des) > 0:
                f.write(str(128) + '\n')
                f.write(str(len(kp)) + '\n')
                for j in range(len(des)):
                    #locs_str = str(int(kp[j].pt.x)) + ' ' + str(int(kp[j].pt.y)) + ' ' + str(int(kp[j].angle)) + ' ' + str(int(kp[j].scale)) + ' ' + str(int(kp[j].octave)) + ' ' # bug
                    locs_str = '0 0 0 0 0 '
                    descs_str = " ".join([str(int(value)) for value in des[j]])
                    all_strs = locs_str + descs_str
                    f.write(all_strs + '\n')
                f.close()
                 
            print "%d(%d), %s, desc: %d" %(i+1, len(img_names), line, len(des))


if __name__ == '__main__':

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()

    parts = 10
    txt_path = '../../data/oxford.txt'
    db_dir = '/home/yuanyong/datasets/oxford'
    save_dir = '../opencv_sifts/'

    with open(txt_path, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    blocks = split_list(content, wanted_parts = parts)
    for i in xrange(0, parts):
        pool.apply_async(cpu_task, args=(blocks[i], db_dir, save_dir,))
    pool.close()
    pool.join()
