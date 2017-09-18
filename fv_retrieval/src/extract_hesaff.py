#!/usr/bin/env python
# encoding: utf-8
# Author: yongyuan.name

import os
import multiprocessing
from multiprocessing import Process, freeze_support, Pool


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]


def cpu_task(img_names, bbin, save_dir):
    for i, line in enumerate(img_names):
        img_path = os.path.join('/home/yuanyong/datasets/oxford', line)
        cmd = bbin + ' ' + img_path + ' ' + save_dir
        os.system(cmd) # returns the exit status
        print "%d(%d), %s" %(i+1, len(img_names), line)


if __name__ == '__main__':

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()

    parts = 50
    bbin = '/home/yuanyong/cpp/hesaff/hesaff'
    txt_path = '/raid/yuanyong/yael_fv/oxford.txt'
    save_dir = '/raid/yuanyong/yael_fv/oxford_hesaff_sift/'

    with open(txt_path, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    blocks = split_list(content, wanted_parts = parts)

    for i in xrange(0, parts):
        pool.apply_async(cpu_task, args=(blocks[i],bbin,save_dir,))
    pool.close()
    pool.join()
