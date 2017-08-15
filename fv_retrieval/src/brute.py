#!/usr/bin/env python
# encoding: utf-8
# Author: yongyuan.name

import os
import sys
import glob
import h5py
import timeit
import cv2
import numpy as np
from shutil import copyfile
from PIL import Image
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA


def run_feature_processing_pipeline(features, d=128, whiten=True, copy=False, params=None):
    features = normalize(features, copy=copy)

    # Whiten and reduce dimension
    if params:
        pca = params['pca']
        features = pca.transform(features)
    else:
        pca = PCA(n_components=d, whiten=whiten, copy=copy)
        features = pca.fit_transform(features)
        params = {'pca': pca}

    # Normalize
    features = normalize(features, copy=copy)

    return features, params


def normalize(x, copy=False):
    """
    A helper function that wraps the function of the same name in sklearn.
    This helper handles the case of a single column vector.
    """
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1, -1), copy=copy))
        #return np.squeeze(x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis])
    else:
        return sknormalize(x, copy=copy)
        #return x / np.sqrt((x ** 2).sum(-1))[..., np.newaxis]


def query_images(groundtruth_dir, image_dir, dataset, names, cropped=True):
    """
    Extract features from the Oxford or Paris dataset.
    :param str groundtruth_dir:
        the directory of the groundtruth files (which includes the query files)
    :param str image_dir:
        the directory of dataset images
    :param str dataset:
        the name of the dataset, either 'oxford' or 'paris'
    :param bool cropped:
        flag to optionally disable cropping
    :yields Image img:
        the Image object
    :yields str query_name:
        the name of the query
    """
    imgs = []
    query_names = []
    fake_query_names = []
    feats_crop = []

    for f in glob.iglob(os.path.join(groundtruth_dir, '*_query.txt')):
        fake_query_name = os.path.splitext(os.path.basename(f))[0].replace('_query', '')
        fake_query_names.append(fake_query_name)

        query_name, x, y, w, h = open(f).read().strip().split(' ')

        if dataset == 'oxford':
            query_name = query_name.replace('oxc1_', '')
            query_names.append('%s.jpg' % query_name)
        img = cv2.imread(os.path.join(image_dir, '%s.jpg' % query_name), 1) # BGR

        if cropped:
            x, y, w, h = map(float, (x, y, w, h))
            x, y, w, h = map(lambda d: int(round(d)), (x, y, w, h))
        else:
            x, y, w, h = (0, 0, img.shape[1], img.shape[0])
        idx = names.index(query_name+'.jpg')
        img = img[y:y+h, x:x+w]
        feat = feats[idx]
        feats_crop.append(feat)
        imgs.append(img)
    return imgs, feats_crop, query_names, fake_query_names


def compute_cosin_distance(Q, feats, names):
    """
    feats and Q: L2-normalize, n*d
    """
    dists = np.dot(Q, feats.T)
    idxs = np.argsort(dists)[::-1]
    rank_dists = dists[idxs]
    rank_names = [names[k] for k in idxs]
    return (idxs, rank_dists, rank_names)

def compute_euclidean_distance(Q, feats, names, k = None):
    if k is None:
        k = len(feats)

    dists = ((Q - feats)**2).sum(axis=1)
    idx = np.argsort(dists) 
    dists = dists[idx]
    rank_names = [names[k] for k in idx]

    return (idx[:k], dists[:k], rank_names)
    

def simple_query_expansion(Q, data, inds, top_k=10):
    """
    Get the top-k closest vectors, average and re-query
    :param ndarray Q:
        query vector
    :param ndarray data:
        index data vectors
    :param ndarray inds:
        the indices of index vectors in ascending order of distance
    :param int top_k:
        the number of closest vectors to consider
    :returns ndarray idx:
        the indices of index vectors in ascending order of distance
    :returns ndarray dists:
        the squared distances
    """
    Q += data[inds[:top_k], :].sum(axis=0)
    return normalize(Q)


def reranking(Q, data, inds, names, top_k = 50):
    vecs_sum = data[0, :]
    for i in range(1, top_k):
        vecs_sum += data[inds[i], :]
    vec_mean = vecs_sum/float(top_k)
    Q = normalize(Q - vec_mean)
    for i in range(top_k):
        data[i, :] = normalize(data[i, :] - vec_mean)
    sub_data = data[:top_k]
    sub_idxs, sub_rerank_dists, sub_rerank_names = compute_cosin_distance(Q, sub_data, names[:top_k])
    names[:top_k] = sub_rerank_names
    return names


if __name__ == '__main__':

    gt_files = '/home/yuanyong/datasets/gt_files'
    dir_images = '/home/yuanyong/datasets/oxford'
 
    # query expansion
    do_QE = True
    topK = 5
    # crop
    do_crop = False
    # reduced dim
    do_pca = False
    redud_d = 128
    do_rerank = False

    # load all features
    start = timeit.default_timer()
    h5f = h5py.File('./oxford_gmm_root_32/fisher_8192.h5', 'r')
    feats = h5f['feats']
    names = list(h5f['names'])
    stop = timeit.default_timer()
    print "load time: %f seconds\n" % (stop - start)

    print "number samples: %d, dimension: %d, number names: %d" %(feats.shape[0], feats.shape[1], len(names))

    # L2-normalize features
    feats = normalize(feats, copy=False)

    # PCA reduce dimension
    if do_pca:
        #import pickle
        #whitening_params = {}
        #if os.path.isfile('../model/pca_model.pkl'):
        #    with open( '../model/pca_model.pkl' , 'rb') as f:
        #        whitening_params['pca'] = pickle.load(f)
        _, whitening_params = run_feature_processing_pipeline(feats, d=redud_d, copy=True)
        feats, _ = run_feature_processing_pipeline(feats, params=whitening_params)

    imgs, query_feats, query_names, fake_query_names = query_images(gt_files, dir_images, 'oxford', names, do_crop)

    #print query_names    
    aps = []
    rank_file = 'tmp.txt'
    for i, query in enumerate(query_names):
        Q = query_feats[i]

        if do_pca:
            Q, _ = run_feature_processing_pipeline([Q], params=whitening_params)       
            Q = np.squeeze(Q.astype(np.float32))
  
        idxs, rank_dists, rank_names = compute_cosin_distance(Q, feats, names)
        #idxs, rank_dists, rank_names = compute_euclidean_distance(Q, feats, names)

        if do_QE:
            Q = simple_query_expansion(Q, feats, idxs, top_k=topK)
            idxs, rank_dists, rank_names = compute_cosin_distance(Q, feats, names)
            #idxs, rank_dists, rank_names = compute_euclidean_distance(Q, feats, names)

        if do_rerank:
            rank_names = reranking(Q, feats, idxs, rank_names, top_k = 50)
 
        # write rank names to txt
        f = open(rank_file, 'w')
        f.writelines([name.split('.jpg')[0] + '\n' for name in rank_names])
        f.close()

        # compute mean average precision
        gt_prefix = os.path.join(gt_files, fake_query_names[i])
        cmd = './compute_ap %s %s' % (gt_prefix, rank_file)
        ap = os.popen(cmd).read()

        os.remove(rank_file)

        aps.append(float(ap.strip()))

        print "%s, %f" %(query, float(ap.strip()))

    print
    print "mAP: %f" % np.array(aps).mean()
