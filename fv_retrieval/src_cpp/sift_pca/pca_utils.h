#ifndef PCA_UTILS_H
#define PCA_UTILS_H

#include <stdio.h>
#include <opencv2/opencv.hpp>

namespace cvtk {

class PCAUtils {
 public:
  static PCAUtils* instance_;

  ~PCAUtils() {
  }

  static PCAUtils* getInstance() {
    static PCAUtils inst;
    return &inst;
  }

  void loadModel(const std::string& filename);

  void reduceDim(const float* data, int num, int dim, cv::Mat& reduceMat);
  void reduceDim(const cv::Mat& mat, cv::Mat& reduceMat);
  cv::Mat reduceDim(const cv::Mat& mat);

 private:

  void convertToMat(const float* data, int num, int dim, cv::Mat& mat);
  cv::PCA pca_;
};

}

#endif /* PCA_UTILS_H */
