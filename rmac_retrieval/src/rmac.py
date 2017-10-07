#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib as mpl
mpl.use('Agg')
import os, sys
import numpy as np
import caffe

from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA


def normalize(x, copy=False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
        #return np.squeeze(x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis])
    else:
        return sknormalize(x, copy=copy)
        #return x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis]


def compute_crow_spatial_weight(X, a=2, b=2):
    """
    Given a tensor of features, compute spatial weights as normalized total activation.
    Normalization parameters default to values determined experimentally to be most effective.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :param int a:
        the p-norm
    :param int b:
        power normalization
    :returns ndarray:
        a spatial weight matrix of size (height, width)
    """
    S = X.sum(axis=0)
    z = (S**a).sum()**(1./a)
    return (S / z)**(1./b) if b != 1 else (S / z)


def pack_regions_for_network( all_regions):
    n_regs = np.sum([len(e) for e in all_regions])
    # print all_regions[0]
    R = np.zeros((n_regs, 5), dtype=np.float32)
    cnt = 0
    # There should be a check of overflow...
    for i, r in enumerate(all_regions):
        try:
            R[cnt:cnt + r.shape[0], 0] = i
            R[cnt:cnt + r.shape[0], 1:] = r
            cnt += r.shape[0]
        except:
            continue
    assert cnt == n_regs
    R = R[:n_regs]
    # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
    R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
    R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
    # print R[3]
    return R


def get_rmac_region_coordinates(H,W,L):
    # Almost verbatim from Tolias et al Matlab implementation.
    # Could be heavily pythonized, but really not worth it...
    # Desired overlap of neighboring regions
    ovr = 0.4
    # Possible regions for the long dimension
    steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
    w = np.minimum(H, W)

    b = (np.maximum(H, W) - w) / (steps - 1)
    # steps(idx) regions for long dimension. The +1 comes from Matlab
    # 1-indexing...
    idx = np.argmin(np.abs(((w ** 2 - w * b) / w ** 2) - ovr)) + 1

    # Region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx
    elif H > W:
        Hd = idx

    regions_xywh = []
    for l in range(1, L + 1):
        wl = np.floor(2 * w / (l + 1))
        wl2 = np.floor(wl / 2 - 1)
        # Center coordinates
        if l + Wd - 1 > 0:
            b = (W - wl) / (l + Wd - 1)
        else:
            b = 0
        cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
        # Center coordinates
        if l + Hd - 1 > 0:
            b = (H - wl) / (l + Hd - 1)
        else:
            b = 0
        cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

        for i_ in cenH:
            for j_ in cenW:
                regions_xywh.append([j_, i_, wl, wl])

    # print "regions number: %d" % len(regions_xywh)
    # Round the regions. Careful with the borders!
    for i in range(len(regions_xywh)):
        for j in range(4):
            regions_xywh[i][j] = int(round(regions_xywh[i][j]))
        if regions_xywh[i][0] + regions_xywh[i][2] > W:
            regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
        if regions_xywh[i][1] + regions_xywh[i][3] > H:
            regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
    return np.array(regions_xywh).astype(np.float32)


def get_rmac_features(X, R):
    # test crow + rmac
    #S = compute_crow_spatial_weight(X)
    #X = X * S

    nfeat = []
    # full feature map
    #feat = np.max(np.max(X, axis=2), axis=1)
    feat_full = X.max(1).max(-1)
    # concat full feature and region
    #nfeat.append(feat_full)
    # regions map
    for r in R:
        newX = X[:, int(r[2]):int(r[4]), int(r[1]):int(r[3])]
        #feat = np.max(np.max(newX,axis=2), axis=1)
        feat = newX.max(1).max(-1)
        #feat=apply_crow_aggregation(newX)
        #feat=feat.reshape(1,-1)
        #feat = normalize(feat, norm='l2')
        nfeat.append(feat)
    #nfeat = np.concatenate(nfeat)
    nfeat = np.array(nfeat)
    # sum pooling
    nfeat = nfeat.sum(axis=0) 
    #nfeat = normalize(nfeat, copy=False)
    #feat_full = normalize(np.array(feat_full), copy=True)
    nfeat = np.concatenate([feat_full, nfeat])
    #nfeat = np.sqrt(nfeat)
    #nfeat = nfeat.reshape(1,-1)
    nfeat = normalize(nfeat)
    return nfeat


def apply_rmac_aggregation(X):
    """
    :param X:
     3d tensor of activations with dimensions (channels, height, width)
    :return ndarray:
     rmac aggregated global image feature
    """
    L = 3
    k, h, w = X.shape
    all_regions = []
    all_regions.append(get_rmac_region_coordinates(h, w, L))
    R = pack_regions_for_network(all_regions)
    feat = get_rmac_features(X, R)
    return feat


def apply_maxpooling_aggregation(self, tensor_3d):
    """
    max pooling aggregation: C*H*W
    """
    return tensor_3d.max(1).max(-1)

