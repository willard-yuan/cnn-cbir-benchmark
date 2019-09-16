## Benchmark for Image Retrieval (BKIR)

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](../LICENSE)

This project tries to build a benchmark for image retrieval, particully for Instance-level image retrieval.

## Methods

The following methods are evaluated on [Oxford Building dataset](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/). The evaluation adopts mean Average Precision (mAP), which is computed using the code provided by [compute_ap.cpp](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp).

method | feature |  mAP (best) | status | evalute code
:---:|:---:|:---:|:---:|:---:
fc_retrieval | CNN | 60.2% | finished | [fc_retrieval](https://github.com/willard-yuan/cnn-cbir-benchmark/tree/master/fc_retrieval)
rmac_retrieval | [RMAC](https://arxiv.org/abs/1511.05879) | 75.7%(256d, crop, qe) | finished | [rmac_retrieval](https://github.com/willard-yuan/cnn-cbir-benchmark/tree/master/rmac_retrieval)
crow_retrieval | [CROW](https://arxiv.org/pdf/1512.04065.pdf) | 72.8%(256d, crop, qe) | finished | [crow_retrieval](https://github.com/willard-yuan/cnn-cbir-benchmark/tree/master/crow_retrieval)
fv_retrieval | SIFT | 67.29% | finished | [fv_retrieval](https://github.com/willard-yuan/cnn-cbir-benchmark/tree/master/fv_retrieval)
vlad_retrieval | SIFT | 63.13% | finished | [vlad_retrieval](https://github.com/willard-yuan/cnn-cbir-benchmark/tree/master/vlad_retrieval)
fv_retrieval | [SOSNet](https://github.com/scape-research/SOSNet) | 50.73% | ongoing | -
vlad_retrieval | [SOSNet](https://github.com/scape-research/SOSNet) | - | ongoing | -

the methods on above have the following characteristics:

- **Low dimension**
- **Time - tested**, and are dimanstracted effectively
- **Used in industry**

## Contribution

If you are interested in this project, feel free to contribute your code. Only Python and C++ code are accepted.
