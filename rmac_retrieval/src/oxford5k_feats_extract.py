#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Extract features offline, without cropping.


import cv2
import matplotlib as mpl
mpl.use('Agg')
import caffe
import multiprocessing
from multiprocessing import Process, freeze_support, Pool
import sys, os, scipy, h5py, glob
import numpy as np
from PIL import Image
from extract_features import load_img, format_img_for_vgg, opencv_format_img_for_vgg, extract_raw_features, extract_multi_raw_features
from rmac import apply_rmac_aggregation

def save_results_image(rank_dists, rank_names, dir_images, save_directory):
    for j, tmp_dist in enumerate(rank_dists):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        copyfile(os.path.join(dir_images, rank_names[j]), os.path.join(save_directory, str(j) + rank_names[j]))

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def gpu_task(prototxt, caffemodel, layer, path_images, out, gpu=0):

    num_images = len(path_images)
    h5f = h5py.File(out, 'w')

    # set gpu card
    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    # init NN
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.forward()

    features = []
    image_names = []
    for i, path in enumerate(path_images):
        print "%d(%d), %s"%((i+1), num_images, os.path.basename(path))
        d = opencv_format_img_for_vgg(path, False)
        # extract feature
        feat = extract_raw_features(net, layer, d)
        #feat = extract_multi_raw_features(net, layer, d)
        # apply maxpooling
        #feat = apply_maxpooling_aggregation(feat)
        #feat = apply_crow_aggregation(feat)
        feat = apply_rmac_aggregation(feat)
        features.append(feat)
        image_names.append(os.path.basename(path))
    features = np.array(features)
    h5f['feats'] = features
    h5f['names'] = image_names
    h5f.close()
    print "gpu %d task has finished..." % (int(gpu))

if __name__ == '__main__':

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()

    layer = 'pool5'
    gpusID = [0,1,2,3,4,5,6,7]
    parts = len(gpusID)
    network = 'VggNet'

    dir_images = '/home/yuanyong/datasets/oxford/*'
    path_images = [os.path.join(dir_images, f) for f in sorted(glob.glob(dir_images))] #if f.endswith('.jpg')]

    # VggNet
    prototxt = '/home/yuanyong/models/pool5.prototxt'
    caffemodel = '/home/yuanyong/models/vgg.model'

    out = './feats/oxford'

    out_files = []
    for i in xrange(parts):
        out_files.append(os.path.join(out, str(i) + '.h5'))

    blocks = split_list(path_images, wanted_parts = parts)

    for i in xrange(0, parts):
        pool.apply_async(gpu_task, args = (prototxt, caffemodel, layer, blocks[i], out_files[i], gpusID[i],))
    pool.close()
    pool.join()
