
import os
import glob
import numpy as np
import cv2
import caffe
import scipy
from functools import partial
from feats_extractor import format_img_for_vgg, extract_raw_features, opencv_format_img_for_vgg
from crow import normalize, save_spatial_weights_as_jpg, compute_crow_spatial_weight, compute_crow_channel_weight

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="source image path")
args = parser.parse_args()

def save_spatial_weights_as_jpg(S, path='.', filename='crow_sw', size=None):
    img = scipy.misc.toimage(S)
    if size is None:
        size = (S.shape[1] * 32, S.shape[0] * 32)
    img = img.resize(size, cv2.INTER_CUBIC)
    img.save(os.path.join(path, '%s.jpg' % str(filename)))

def loadImage(path, resize = True, height = 640, width = 480):
    """
    Load image from image path and do preprocessing
    """
    img = cv2.imread(path, 1) # BGR
    if resize == True:
        img = cv2.resize(img, (width, height), cv2.INTER_LINEAR)
    d = np.float32(img)
    d -= np.array((104.00698793,116.66876762,122.67891434))
    return (d.transpose((2,0,1)), img) # (H, W, C) to (C, H, W)

if __name__ == '__main__':

    caffe.set_device(6)
    caffe.set_mode_gpu()

    num_largest = 7
    layer = 'pool5'
    prototxt = '../models/caffe.proto'
    caffemodel = '../models/vgg.model'

    save_path = '../heatmap/'

    # Load networks
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.blobs['data'].reshape(1, 3, 640, 480) # data blob is N x C x H x W
    net.forward()

    # extract query feature
    (d, img) = loadImage(args.img_path, False)
    Q = extract_raw_features(net, layer, d)
    S = compute_crow_spatial_weight(Q)

    indices = (-S).argpartition(num_largest, axis=None)[:num_largest]
    xs, ys = np.unravel_index(indices, S.shape)
    print d.shape
    save_spatial_weights_as_jpg(S, save_path, 'spatial_weights', size = (d.shape[2], d.shape[1]))
    for i in xrange(num_largest):
        x, y = ((xs[i]+1)*32, (ys[i]+1)*32)
        img_roi = img[x-1-128:x-1+128, y-1-128:y-1+128]
        cv2.imwrite(os.path.join(save_path, str(i) + '.jpg'), img_roi)
    #C = compute_crow_channel_weight(Q)
    #Q = Q * S
    #save_spatial_weights_as_jpg(C, save_path, 'channel_weight') ## error
    #Q = Q.sum(axis=(1, 2))*C
    #print Q
