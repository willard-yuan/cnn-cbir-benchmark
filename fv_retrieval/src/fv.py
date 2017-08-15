#!/usr/bin/env python
# encoding: utf-8
# Author: yongyuan.name

import os,h5py
import numpy as np
from yael import ynumpy


h5f = h5py.File('./oxford_gmm_root_32/fisher_8192.h5', 'w')
weights = np.load('./oxford_gmm_root_32/w.gmm.npy')
mu = np.load('./oxford_gmm_root_32/mu.gmm.npy')
sigma = np.load('./oxford_gmm_root_32/sigma.gmm.npy')
mean = np.load('./oxford_gmm_root_32/mean.gmm.npy')
pca_transform = np.load('./oxford_gmm_root_32/pca_transform.gmm.npy')

gmm = [weights, mu, sigma]

# 
txt_path = '/home/yuanyong/py/fv_retrieval/oxford.txt'
with open(txt_path, 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content]

# 
features = []
image_names = []
for i, line in enumerate(content):
    img_name = os.path.basename(line)
    print "%d(%d): %s" %(i+1, len(content), img_name)
    hesaff_path = os.path.join('/home/yuanyong/py/fv_retrieval/oxford_hesaff_sift', os.path.splitext(os.path.basename(line))[0] + '.hesaff.sift')
    hesaff_info = np.loadtxt(hesaff_path)
    if hesaff_info.shape[0] == 0:
        hesaff_info = np.zeros((1, 133), dtype = 'float32')
    elif hesaff_info.shape[0] > 0 and len(hesaff_info.shape) == 1:
        hesaff_info = hesaff_info.reshape([1, 133])

    image_desc = np.sqrt(hesaff_info[:, 5:])
    #n_sifts = image_desc.shape[0]
    #for i in range(n_sifts):
    #    if np.linalg.norm(image_desc[i], ord=2) == 0.0:
    #        continue
    #    image_desc[i] = image_desc[i]/np.linalg.norm(image_desc[i], ord=2)

    # root-sift
    #n_sifts = image_desc.shape[0]
    #for i in range(n_sifts):
    #    if np.linalg.norm(image_desc[i], ord=1) == 0.0:
    #        continue
    #    image_desc[i] = np.sqrt(image_desc[i]/np.linalg.norm(image_desc[i], ord=1))

    #n_sifts = image_desc.shape[0]
    #for i in range(n_sifts):
    #    image_desc[i] = np.sign(image_desc[i]) * np.log(1.0 + np.abs(image_desc[i]))

    # apply the PCA to the image descriptor
    image_desc = np.dot(image_desc - mean, pca_transform)
    image_desc = image_desc.astype(np.float32)

    # compute the Fisher vector, using only the derivative w.r.t mu
    fv = ynumpy.fisher(gmm, image_desc, include = ['mu','sigma'])
    features.append(fv)
    image_names.append(img_name)

# make one matrix with all FVs
features = np.vstack(features)

# normalizations are done on all descriptors at once

# power-normalization
features = np.sign(features) * np.abs(features) ** 0.5

# L2 normalize
#norms = np.sqrt(np.sum(image_fvs ** 2, 1))
#image_fvs /= norms.reshape(-1, 1)

# save feats
print "number samples: %d, dimension: %d, number names: %d" %(features.shape[0], features.shape[1], len(image_names))
h5f['feats'] = features
h5f['names'] = image_names
h5f.close()
