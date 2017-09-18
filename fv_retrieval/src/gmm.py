#!/usr/bin/env python
# encoding: utf-8
# Author: yongyuan.name

import os
import numpy as np
from yael import ynumpy

txt_path = '/home/yuanyong/py/fv_retrieval/oxford.txt'
sift_dir = '/home/yuanyong/py/fv_retrieval/oxford_hesaff_sift'

with open(txt_path, 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content]


all_desc = []
for i, line in enumerate(content):
    print "%d(%d): %s" %(i+1, len(content), line)
    hesaff_path = os.path.join(sift_dir, os.path.splitext(os.path.basename(line))[0] + '.hesaff.sift')
    hesaff_info = np.loadtxt(hesaff_path, skiprows=2)
    if hesaff_info.shape[0] == 0:
       continue
    elif hesaff_info.shape[0] > 0 and len(hesaff_info.shape) == 1:
        desc = hesaff_info[5:]
        all_desc.append(desc)
    elif hesaff_info.shape[0] > 0 and len(hesaff_info.shape) > 1:
        desc = hesaff_info[:, 5:]
        all_desc.append(desc)


# make a big matrix with all image descriptors
all_desc = np.sqrt(np.vstack(all_desc))
#n_sifts = all_desc.shape[0]
#for i in range(n_sifts):
#    if np.linalg.norm(all_desc[i], ord=2) == 0.0:
#        continue
#    all_desc[i] = all_desc[i]/np.linalg.norm(all_desc[i], ord=2)

# sift: root-sift
#n_sifts = all_desc.shape[0]
#for i in range(n_sifts):
    #if np.linalg.norm(all_desc[i], ord=1) == 0.0:
    #    continue
    #all_desc[i] = np.sqrt(all_desc[i]/np.linalg.norm(all_desc[i], ord=1))

# sift: sign(x)log(1 + |x|)
#n_sifts = all_desc.shape[0]
#for i in range(n_sifts):
#    all_desc[i] = np.sign(all_desc[i]) * np.log(1.0 + np.abs(all_desc[i]))


k = 128
n_sample = 256 * 1000

# choose n_sample descriptors at random
np.random.seed(1024)
sample_indices = np.random.choice(all_desc.shape[0], n_sample)
sample = all_desc[sample_indices]

# until now sample was in uint8. Convert to float32
sample = sample.astype('float32')

# compute mean and covariance matrix for the PCA
mean = sample.mean(axis = 0)
sample = sample - mean
cov = np.dot(sample.T, sample)

# compute PCA matrix and keep only 64 dimensions
eigvals, eigvecs = np.linalg.eig(cov)
perm = eigvals.argsort()                   # sort by increasing eigenvalue
pca_transform = eigvecs[:, perm[96:128]]   # eigenvectors for the 64 last eigenvalues

# transform sample with PCA (note that numpy imposes line-vectors,
# so we right-multiply the vectors)
sample = np.dot(sample, pca_transform)

# train GMM
print "start train GMM ......."
gmm = ynumpy.gmm_learn(sample, k, nt = 400, niter = 2000, seed = 0, redo = 1, use_weights = True)

np.save("./oxford_gmm_root_32/w.gmm", gmm[0])
np.save("./oxford_gmm_root_32/mu.gmm", gmm[1])
np.save("./oxford_gmm_root_32/sigma.gmm", gmm[2])
np.save("./oxford_gmm_root_32/mean.gmm", mean)
np.save("./oxford_gmm_root_32/pca_transform.gmm", pca_transform)
