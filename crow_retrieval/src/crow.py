# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

# !/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import scipy
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA
from numpy import linalg as LA


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


def compute_crow_channel_weight(X):
    """
    Given a tensor of features, compute channel weights as the
    log of inverse channel sparsity.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        a channel weight vector
    """
    K, w, h = X.shape
    area = float(w * h)
    nonzeros = np.zeros(K, dtype=np.float32)
    for i, x in enumerate(X):
        nonzeros[i] = np.count_nonzero(x) / area

    nzsum = nonzeros.sum()
    for i, d in enumerate(nonzeros):
        nonzeros[i] = np.log(nzsum / d) if d > 0. else 0.

    return nonzeros


def feature_selection(X):
    heatmap = np.zeros([X.shape[1], X.shape[2]])
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
           heatmap[i, j] = LA.norm(X[:, i, j])
    # median value
    #value = np.median(heatmap)
    # mean value
    value = np.mean(heatmap)
    selectMap = heatmap > value
    selectMap = selectMap.astype(float)
    return selectMap

def apply_crow_aggregation(X):
    """
    Given a tensor of activations, compute the aggregate CroW feature, weighted
    spatially and channel-wise.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        CroW aggregated global image feature    
    """
    S = compute_crow_spatial_weight(X)
    C = compute_crow_channel_weight(X)
    #selectMap = feature_selection(X)
    #X = (X * selectMap)  * S
    X = X * S
    X = X.sum(axis=(1, 2))
    return X * C


def apply_maxpooling_aggregation(X):
    """
    max pooling aggregation
    """
    return X.max(1).max(-1)


def apply_ucrow_aggregation(X):
    """
    Given a tensor of activations, aggregate by sum-pooling without weighting.

    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        unweighted global image feature    
    """
    return X.sum(axis=(1, 2))


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


def run_feature_processing_pipeline(features, d=128, whiten=True, copy=False, params=None):
    """
    Given a set of feature vectors, process them with PCA/whitening and return the transformed features.
    If the params argument is not provided, the transformation is fitted to the data.

    :param ndarray features:
        image features for transformation with samples on the rows and features on the columns
    :param int d:
        dimension of final features
    :param bool whiten:
        flag to indicate whether features should be whitened
    :param bool copy:
        flag to indicate whether features should be copied for transformed in place
    :param dict params:
        a dict of transformation parameters; if present they will be used to transform the features

    :returns ndarray: transformed features
    :returns dict: transform parameters
    """
    # Normalize
    features = normalize(features, copy=copy)

    # Whiten and reduce dimension
    if params:
        pca = params['pca']
        features = pca.transform(features)
    else:
        pca = PCA(n_components=d, whiten=whiten, copy=copy)
        features = pca.fit_transform(features)
        params = { 'pca': pca }

    # Normalize
    features = normalize(features, copy=copy)

    return features, params


def save_spatial_weights_as_jpg(S, path='.', filename='crow_sw', size=None):
    """
    Save an image for visualizing a spatial weighting. Optionally provide path, filename,
    and size. If size is not provided, the size of the spatial map is used. For instance,
    if the spatial map was computed with VGG, setting size=(S.shape[0] * 32, S.shape[1] * 32)
    will scale the spatial weight map back to the size of the image.

    :param ndarray S:
        spatial weight matrix
    :param str path:
    :param str filename:
    :param tuple size:
    """
    img = scipy.misc.toimage(S)
    if size is not None:
        img = img.resize(size)

    img.save(os.path.join(path, '%s.jpg' % str(filename)))
