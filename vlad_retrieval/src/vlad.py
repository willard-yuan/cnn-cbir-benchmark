#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Anthor: yongyuan.name

import os,h5py
import numpy as np
from yael import ynumpy

# save feature to h5py
h5f = h5py.File('./models/vlad_128_8192.h5', 'w')
# load PCA and KMeans model
centroids = np.load('./models/centroids_data.npy')
pca_transform = np.load('./models/pca_data.npy')
mean = np.load('./models/mean_data.npy')

#
txt_path = '/home/yuanyong/py/fv_retrieval/oxford.txt'
with open(txt_path, 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content]

# compute VLAD feature
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

    # compute VLAD
    vlad = ynumpy.vlad(centroids, image_desc)  # num_centroids * d
    # L2 normalization
    norms_vlad = np.sqrt(np.sum(vlad ** 2, 1))
    norms_vlad[norms_vlad < 1e-12] = 1e-12
    vlad /= norms_vlad.reshape(-1, 1)
    vlad = vlad.flatten()
    features.append(vlad)
    image_names.append(img_name)

# make one matrix with all FVs
features = np.vstack(features)

# power-normalization, square-rooting normalization
features = np.sign(features) * np.abs(features) ** 0.5

# L2 normalize
norms = np.sqrt(np.sum(features ** 2, 1))
norms[norms < 1e-12] = 1e-12
features /= norms.reshape(-1, 1)

# save feats
print "number samples: %d, dimension: %d, number names: %d" %(features.shape[0], features.shape[1], len(image_names))
h5f['feats'] = features
h5f['names'] = image_names
h5f.close()
