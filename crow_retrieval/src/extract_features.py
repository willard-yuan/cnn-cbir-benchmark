# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os
import caffe
import numpy as np
from PIL import Image
import scipy
import cv2


###################################
# Feature Extraction
###################################

def load_img(path):
    """
    Load the image at the provided path and normalize to RGB.

    :param str path:
        path to image file
    :returns Image:
        Image object
    """
    try:
        img = Image.open(path)
        rgb_img = Image.new("RGB", img.size)
        rgb_img.paste(img)
        return rgb_img
    except:
        return None



def format_img_for_vgg(img):
    """
    Given an Image, convert to ndarray and preprocess for VGG.

    :param Image img:
        Image object
    :returns ndarray:
        3d tensor formatted for VGG
    """
    # Get pixel values
    d = np.array(img, dtype=np.float32) # set each element to float32
    d = d[:,:,::-1] # RGB to BGR

    # Subtract mean pixel values of VGG training set
    # B: 104.00698793 G: 116.66876762 R: 122.67891434
    d -= np.array((104.00698793,116.66876762,122.67891434))
    # ResNet
    #d -= np.array((103.06262380097594, 115.90288257386003, 123.15163083845863))
   
    return d.transpose((2,0,1)) # (H, W, C) to (C, H, W)


def opencv_format_img_for_vgg(path, resize = False):
    """
    opencv test
    """
    img = cv2.imread(path, -1) # BGR
    if resize == True:
        img = cv2.resize(img, (min(max(img.shape[1], 224), 30000), min(max(img.shape[0], 224), 30000)), interpolation = cv2.INTER_LINEAR) # (width=480, height=640)
    d = np.float32(img)
    # VggNet
    d -= np.array((104.00698793,116.66876762,122.67891434))
    # ResNet
    #d -= np.array((103.06262380097594, 115.90288257386003, 123.15163083845863)) 
    return d.transpose((2,0,1)) # (H, W, C) to (C, H, W)


def extract_raw_features(net, layer, d):
    """
    Extract raw features for a single image.
    """
    # Shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *d.shape)
    net.blobs['data'].data[...] = d
    # run net and take argmax for prediction
    net.forward()
    return net.blobs[layer].data[0]


def extract_multi_raw_features(net, layers,  d):
    net.blobs['data'].reshape(1, *d.shape)
    net.blobs['data'].data[...] = d
    net.forward()
    tmp_feats = []
    for i, layer in enumerate(layers):
        tmp_feats.append(net.blobs[layer].data[0])
    return np.concatenate(tmp_feats)


def get_imlist(path):
    """
    Returns a list of filenames for
    all jpg images in a directory. 
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

def load_batch_images( list_batch ):
    ima_batch = []
    for name in list_batch:
        ima_batch.append( opencv_format_img_for_vgg( name ) )
    return ima_batch

def extract_batch_raw_features(net, layer, batch_images, batch_size):
    """
    Extract raw features for a single image.
    """
    # Shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(batch_size, *batch_images[0].shape)
    
    for k, ima in enumerate(batch_images):
        net.blobs['data'].data[k,...] = ima    

    # forward image through the net
    net.forward()
    return net.blobs[layer].data.copy().astype(np.float32)

if __name__ == '__main__':
    caffe.set_device(1)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()
    images = get_imlist('/media/disk1/yuanyong/data_sets/10w_images/')
    layer = 'pool5'
    prototxt = '/home/yuanyong/python/crow/vgg/VGG_ILSVRC_16_pool5.prototxt'
    caffemodel = '/home/yuanyong/models/VGG_ILSVRC_16_layers.caffemodel'
    out = '/media/disk1/yuanyong/crow/pool5_10w'

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.forward()

    if not os.path.exists(out):
        os.makedirs(out)

    num_images = len(images)

    for i, path in enumerate(images):
        print "%d/%d, %s" %((i+1), num_images, os.path.basename(path))
        
        if os.path.splitext(os.path.basename(path))[0][-6:] != 'single': 
            img = load_img(path)

            # Skip if the image failed to load
            if img is None:
                print path
                continue

            d = format_img_for_vgg(img)
            X = extract_raw_features(net, layer, d)

            filename = os.path.splitext(os.path.basename(path))[0]
            np.save(os.path.join(out, filename), X)
