## CROW image retrieval

**CROW** means using CROW layer as image feature.

## Poject structure

```sh
➜  crow_retrieval git:(master) ✗ tree -L 2
.
├── README.md
├── feats
├── model
│   └── pool5_vgg16.prototxt
├── src
│   ├── brute.py
│   ├── extract_features.py
│   └── oxford5k_feats_extract.py
└── tools
    └── compute_ap.cpp
```

## Build evaluation script

The official C++ program provided by the Oxford group, compute_ap.cpp, for computing the mean average precision (mAP) on the retrieval benchmark is provided in this repository for convenience. It is modified to add an explicit include so that it can be compiled everywhere.

```sh
g++ -O compute_ap.cpp -o compute_ap
```

## CNN Model

model | Dataset used to fine-tuning | layer | mAP
:---:|:---:|:---:|:---:|
[VGG_ILSVRC_16_layers.caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) | - | fc6 | 45.9%
