## FC image retrieval

**FC** means using fully-connected layer as image feature. The FC image retrieval tries to provide a baseline for layer selection for image retrieval. For more details, visit to the [post](http://yongyuan.name/blog/layer-selection-and-finetune-for-cbir.html).

## Poject structure

```sh
➜  fc_retrieval git:(master) ✗ tree -L 2
.
├── README.md
├── feats
├── model
│   ├── pool5_neural.prototxt
│   └── pool5_vgg16.prototxt
├── src
│   ├── brute.py
│   ├── extract_features.py
│   └── oxford5k_feats_extract.py
└── tools
    └── compute_ap
```

## CNN Model

model | Dataset used to fine-tuning | layer | mAP
:---:|:---:|:---:|:---:|
[neural.caffemodel](http://pan.baidu.com/s/1i44RRgx) | [landmark dataset](https://pan.baidu.com/s/1mit6Izm) (baiduyun code: 8hjn)| fc6 | 60.2%
[VGG_ILSVRC_16_layers.caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) | - | fc6 | 45.9%
