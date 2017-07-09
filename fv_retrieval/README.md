## FV image retrieval

**FV** means using Fisher Vector as image feature. The FV is a powerful representation for image retrieval. For more details about Fisher Vector, visit to the [post](http://yongyuan.name/blog/CBIR-BoF-VLAD-FV.html).

Since the dimension of Fisher Vector is unusally very high, we must make a compromise between precision and memory. It's recommended to reduce the dimension of SIFT to 32 dimension or 64 dimension. In real world applicaiton, the dimension of Fisher Vector is recommended as 8192, so the number of kernels of GMM is 128. The mAP on Oxford building dataset of Fisher Vector is as follows:

SIFT dimension | the number of kernels of GMM | Fisher Vector dimension | mAP | mAP (QE) |
:---:|:---:|:---:|:---:|:---:|
32 | 128 | 8192 | 58.67% | 67.29%