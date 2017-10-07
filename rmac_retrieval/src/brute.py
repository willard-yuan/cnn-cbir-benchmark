#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import glob
import h5py
import timeit
import cv2
import numpy as np
from shutil import copyfile
from PIL import Image

from extract_features import load_img, format_img_for_vgg, opencv_format_img_for_vgg, extract_raw_features, extract_multi_raw_features

from sklearn.preprocessing import normalize as sknormalize
from rmac import apply_rmac_aggregation
from sklearn.decomposition import PCA

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


def save_results_image(rank_dists, rank_names, dir_images, save_directory):
    """
    save result images
    """
    for j, tmp_dist in enumerate(rank_dists):
        if 0.9999 > tmp_dist > 0.9:
            if not os.path.exists(directory):
                os.makedirs(save_directory)
            copyfile(os.path.join(dir_images, rank_names[j]),
                     os.path.join(directory, str(j) + '_a_' + rank_names[j]))
        elif 0.9 >= tmp_dist > 0.8:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            copyfile(os.path.join(dir_images, rank_names[j]),
                     os.path.join(save_directory, str(j) + '_b_' + rank_names[j]))

        if os.path.exists(save_directory):
            copyfile(os.path.join(dir_images, rank_names[0]), os.path.join(directory, rank_names[0]))


def query_images(groundtruth_dir, image_dir, dataset, cropped=True):
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

    import caffe
    modelDir = "/home/yuanyong/models"
    #ResNet
    MODEL = "vgg.model"
    PROTO = "pool5.prototxt"
    caffemodel = os.path.join(modelDir, MODEL)
    prototxt = os.path.join(modelDir, PROTO)
    # set gpu card
    layer = 'pool5'
    caffe.set_device(5)
    caffe.set_mode_gpu()
    # init NN
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.forward()

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
        img = img[y:y+h, x:x+w]
        d = np.float32(img)
        # VggNet
        d -= np.array((104.00698793, 116.66876762, 122.67891434))
        d = d.transpose((2, 0, 1))
        feat = extract_raw_features(net, layer, d)
        feat = apply_rmac_aggregation(feat)
        # L2-normalize feature
        feat = normalize(feat, copy=False)
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


def load_files(files):
    """
    Function: load features from files
    files: list
    """
    h5fs = {}
    for i, f in enumerate(files):
        h5fs['h5f_' + str(i)] = h5py.File(f, 'r')
    feats = np.concatenate([value['feats'] for key, value in h5fs.items()])
    names = np.concatenate([value['names'] for key, value in h5fs.items()])
    return (feats, names)

if __name__ == '__main__':

    gt_files = '/home/yuanyong/datasets/gt_files'
    feats_files = '/home/yuanyong/py/rmac_retrieval/feats/oxford/*'
    dir_images = '/home/yuanyong/datasets/oxford'
 
    # query expansion
    do_QE = False
    topK = 5
    # crop
    do_crop = False
    # reduced dim
    do_pca = False
    do_rerank = False
    redud_d = 256

    # load all features
    start = timeit.default_timer()
    files =  glob.glob(feats_files)
    feats, names = load_files(files)
    stop = timeit.default_timer()
    print "load time: %f seconds\n" % (stop - start)

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

    imgs, query_feats, query_names, fake_query_names = query_images(gt_files, dir_images, 'oxford', do_crop)

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
    print np.array(aps).mean()
