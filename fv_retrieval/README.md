## Fisher Vector for Image Retrieval

### Introduction

**FV** means using Fisher Vector as image feature. The FV is a powerful representation for image retrieval including BoVW and VLAD encoding. For more details about Fisher Vector, please visit to the [post](http://yongyuan.name/blog/CBIR-BoF-VLAD-FV.html).

Since the dimension of Fisher Vector is unusally very high, we must make a compromise between precision and memory. It's strongly recommended to reduce the dimension of SIFT to 32 dimension or 64 dimension. In real world applicaiton, the dimension of Fisher Vector is recommended as 8192, so the number of Guass kernel of GMM is set as 128. The mAP on Oxford building dataset of Fisher Vector is as follows:

Image Size | SIFT dimension | Number of kernels of GMM | Fisher Vector dimension | mAP | mAP (QE) |
:---:|:---:|:---:|:---:|:---:|:---:|
0.5*w x 0.5*h | 32 | 128 | 8192 | 58.67% | 67.29%

Here I choose [Hessian Affine detector with SIFT descriptor](https://github.com/perdoch/hesaff) instead of original SIFT, since the performance of Hessian SIFT is usually better than original SIFT. Besides, to spped up the time of Hessian SIFT descriptors extraction, I scale down the image with 0.5 factor. **As far as I know**, the implemention achieves very promising performance compare with other FV implemention.

### Reimplement the experiment