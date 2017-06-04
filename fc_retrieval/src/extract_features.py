import os
import caffe
import numpy as np
from PIL import Image
import scipy
import cv2


def opencv_format_img_for_vgg(path, resize = False):
    """
    opencv test
    """
    img = cv2.imread(path, -1) # BGR
    if resize == True:
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
    d = np.float32(img)
    d -= np.array((104.00698793,116.66876762,122.67891434))
    return d.transpose((2,0,1)) # (H, W, C) to (C, H, W)


def extract_fc_features(net, layer, d):
    """
    Extract raw features for a single image.
    """
    # Shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *d.shape)
    net.blobs['data'].data[...] = d
    # run net and take argmax for prediction
    net.forward()
    return net.blobs[layer].data[0]

