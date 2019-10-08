## VLAD for Image Retrieval

### Introduction

**VLAD** means using aggregating local descriptors as image feature. The VLAD is a powerful representation for image retrieval including BoVW and FV encoding. For more details about VLAD Vector, please visit to the [post](http://yongyuan.name/blog/CBIR-BoF-VLAD-FV.html).

Since the dimension of VLAD Vector is unusally very high, we must make a compromise between precision and memory. It's strongly recommended to reduce the dimension of SIFT to 32 dimension or 64 dimension. In real world applicaiton, the dimension of VLAD Vector is recommended as 8192, so the number of Guass kernel of GMM is set as 128. The mAP on Oxford building dataset of Fisher Vector is as follows:

Image Size | SIFT dimension | Number of KMeans | VLAD Vector dimension | mAP | mAP (QE) |
:---:|:---:|:---:|:---:|:---:|:---:|
0.5*w x 0.5*h | 32 | 128 | 8192 | 56.39% | 63.13%

Here I choose [Hessian Affine detector with SIFT descriptor](https://github.com/perdoch/hesaff) instead of original SIFT, since the performance of Hessian SIFT is usually better than original SIFT. Besides, to spped up the time of Hessian SIFT descriptors extraction, I scale down the image with 0.5 factor. The normalization is power normalization followed by a L2 normalization. You can also do it with intra-normalization. **As far as I know**, the implemention achieves very promising performance compare with other VLAD implemention.

### Reimplement the experiment
