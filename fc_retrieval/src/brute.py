#!/usr/bin/env python
# encoding: utf-8

import cv2
import matplotlib as mpl
mpl.use('Agg')
import caffe
import os, sys, h5py, timeit, glob
import numpy as np
from shutil import copyfile
from PIL import Image

from sklearn.preprocessing import normalize as sknormalize
from extract_features import extract_fc_features, opencv_format_img_for_vgg
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA

def normalize(x, copy=False):
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
    else:
        return sknormalize(x, copy=copy)


def run_feature_processing_pipeline(features, d=128, whiten=True, copy=False, params=None):
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


def query_images(groundtruth_dir, image_dir, dataset, cropped=True):
    imgs = []
    query_names = []
    fake_query_names = []
    feats_crop = [] 
   
    modelDir = "/home/yuanyong/py/fc_retrieval/model"
    MODEL = "nueral.caffemodel"
    PROTO = "deploy.prototxt"
    caffemodel = os.path.join(modelDir, MODEL)
    prototxt = os.path.join(modelDir, PROTO)

    # set gpu card
    layer = 'fc6'
    caffe.set_device(7)
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
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
        d = np.float32(img)
        # VggNet
        d -= np.array((104.00698793, 116.66876762, 122.67891434))
        d = d.transpose((2, 0, 1))
        feat = extract_fc_features(net, layer, d)
        # L2-normalize feature
        feat = normalize(feat, copy=False)
        #feats_crop.append(feat.tolist())
        feats_crop.append(np.array(feat))
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
    

def simple_query_expansion(Q, data, inds, top_k = 10):
    #Q += data[inds[:top_k], :].sum(axis=0)
       
    # weighted query
    for i in range(top_k):
        Q += (1.0*(top_k-i)/float(top_k))*data[inds[i], :]
        #Q += data[inds[i], :]
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
    h5fs = {}
    for i, f in enumerate(files):
        h5fs['h5f_' + str(i)] = h5py.File(f, 'r')
    feats = np.concatenate([value['feats'] for key, value in h5fs.items()])
    names = np.concatenate([value['names'] for key, value in h5fs.items()])
    return (feats, names)

if __name__ == '__main__':

    gt_files = '/home/yuanyong/datasets/gt_files'
    feats_files = '/home/yuanyong/py/fc_retrieval/feats/*'
    dir_images = '/home/yuanyong/datasets/oxford'
 
    # query expansion
    do_QE = False
    topK = 10
    do_crop = True
    do_pca = True
    do_rerank = True
    redud_d = 128

    # load all features
    start = timeit.default_timer()
    files =  glob.glob(feats_files)
    feats, names = load_files(files)
    print feats.shape
    stop = timeit.default_timer()
    print "load time: %f seconds\n" % (stop - start)

    # L2-normalize features
    feats = normalize(feats, copy=False)

    # PCA reduce dimension
    if do_pca:
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
            Q = simple_query_expansion(Q, feats, idxs, top_k = topK)
            idxs, rank_dists, rank_names = compute_cosin_distance(Q, feats, names)
            #idxs, rank_dists, rank_names = compute_euclidean_distance(Q, feats, names)
        
        if do_rerank:
            rank_names = reranking(Q, feats, idxs, rank_names, top_k = 10)

        # write rank names to txt
        f = open(rank_file, 'w')
        f.writelines([name.split('.jpg')[0] + '\n' for name in rank_names])
        f.close()


        # compute mean average precision
        gt_prefix = os.path.join(gt_files, fake_query_names[i])
        cmd = '../tools/compute_ap %s %s' % (gt_prefix, rank_file)
        ap = os.popen(cmd).read()
        os.remove(rank_file)
        aps.append(float(ap.strip()))
        print "%s, %f" %(query, float(ap.strip()))

    print
    print np.array(aps).mean()
