#include "pca_utils.h"

namespace cvtk {

void PCAUtils::convertToMat(const float* data, int num, int dim, cv::Mat& mat) {
  mat.create(num, dim, CV_32FC1);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      mat.at<float>(i, j) = *(data + i * dim + j);
    }
  }
}

void PCAUtils::loadModel(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  // opencv 3.2
  cv::read(fs.root()["vectors"], pca_.eigenvectors);
  cv::read(fs.root()["values"], pca_.eigenvalues);
  cv::read(fs.root()["mean"], pca_.mean);
}

void PCAUtils::reduceDim(const cv::Mat& mat, cv::Mat& reduceMat) {
  pca_.project(mat, reduceMat);
  // L2 normalize
  /*
  for (int i = 0; i < reduceMat.rows; ++i) {
    cv::Mat normMat = reduceMat.row(i) * reduceMat.row(i).t();
    float denomv = std::max(1e-12, sqrt(normMat.at<float>(0, 0)));
    for (int j = 0; j < reduceMat.cols; ++j) {
      reduceMat.at<float>(i, j) = reduceMat.at<float>(i, j) / denomv;
    }
  }
  */
}

void PCAUtils::reduceDim(const float* data, int num, int dim, cv::Mat& reduceMat) {
  cv::Mat mat;
  convertToMat(data, num, dim, mat);
  reduceDim(mat, reduceMat);
}

cv::Mat PCAUtils::reduceDim(const cv::Mat& mat) {
  cv::Mat reduceMat;
  reduceDim(mat, reduceMat);
  return reduceMat;
}

}
