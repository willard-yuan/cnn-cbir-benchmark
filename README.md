## Benchmark for Image Retrieval (BKIR)

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](../LICENSE)

This project tries to build a benchmark for image retrieval, particully for Instance-level image retrieval.

## Methods

The following methods are evaluated on [Oxford Building dataset](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/). The evaluation adopts mean Average Precision (mAP), which is computed using the code provided by [compute_ap.cpp](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp).

method | feature |  mAP (best) | status | links
:---:|:---:|:---:|:---:|:---:
fc_retrieval | CNN | 60.2% | finished | [fc_retrieval](https://github.com/willard-yuan/cnn-cbir-benchmark/tree/master/fc_retrieval)
rmac_retrieval | CNN | - | finished | coming soon
crow_retrieval | CNN | - | finished | coming soon
fv_retrieval | SIFT | 67.29% | finished | [fv_retrieval](https://github.com/willard-yuan/cnn-cbir-benchmark/tree/master/fv_retrieval)
vlad_retrieval | SIFT | - | ongoing | -

the methods on above have the following characteristics:

- **Low dimension**
- **Time - tested**, and are dimanstracted effectively
- **Used in industry**

## Contribution

If you are interested in this project, feel free to contribute your code. Only Python and C++ code are accepted.