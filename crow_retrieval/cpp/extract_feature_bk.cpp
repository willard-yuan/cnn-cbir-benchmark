// Author: yongyuan.name

#include <string>
#include <vector>
#include <stdio.h>
#include <math.h>

#include "boost/make_shared.hpp"
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>

using namespace cv;
using namespace Eigen; 
using namespace std;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

// Function: Convert 1D array to 3D tensor
void array_to_maps(const float* feature_blob_data, std::vector<MatrixXf> &maps, MatrixXf &sum_maps, int map_height, int map_width, int blob_num_channels){
    float * temp = new float[map_height*map_width];
    for (int i = 0; i < blob_num_channels; i++){
        // memcpy(temp, &feature_blob_data[map_height*map_width*i], map_height*map_width*sizeof(feature_blob_data[0]));
        memcpy(temp, feature_blob_data + map_height*map_width*i, map_height*map_width*sizeof(feature_blob_data[0]));
        MatrixXf eigenX = Map<MatrixXf>(temp, map_width, map_height).transpose();
        sum_maps = sum_maps + eigenX;
        maps.push_back(eigenX);
    }
    delete [] temp;
}

// Function: Compute spatial weight
MatrixXf compute_spatial_weight(const MatrixXf &sum_maps){
    // Square all elements in matrix square_sum_maps
    MatrixXf square_sum_maps = sum_maps.cwiseProduct(sum_maps);
    // Sum all elements of matrix square_sum_maps, root need to improve
    float z = std::sqrt(square_sum_maps.sum());
    // Power nornalization 
    MatrixXf spatial_weight = sum_maps / z;
    spatial_weight = spatial_weight.cwiseSqrt();
    return spatial_weight;
}

// Function: Compute channel weight
void compute_channel_weight(MatrixXf &channel_weight, std::vector<MatrixXf> &maps, int map_height, int map_width, int blob_num_channels){
    // Normalize non-zeros numbers
    float area = 1.0*map_height*map_width;
    SparseMatrix<float> spMat;
    for (int i = 0; i < blob_num_channels; i++){
        spMat = maps[i].sparseView();
        channel_weight(0, i) = spMat.nonZeros()/area;
    }
    // computer spatial weight
    float sum_nonzeros = channel_weight.sum();
    for (int i = 0; i < blob_num_channels; i++){
        if (channel_weight(0, i) > 0.0)
            channel_weight(0, i) = log(sum_nonzeros / channel_weight(0, i));
        else
            channel_weight(0, i) = 0.0;
    }
}

// Function: Generate ImageNet 2012 mean CV:Mat
// Note: The cv:Mat format is ordered in BGR
// B: 104.00698793 G: 116.66876762 R: 122.67891434
cv::Mat mean_mat_generation(int img_height, int img_width){
    std::vector<cv::Mat> pixes_mat;
    Mat B_Mat(img_height, img_width, CV_32FC1, Scalar(104.00698793));
    Mat G_Mat(img_height, img_width, CV_32FC1, Scalar(116.66876762));
    Mat R_Mat(img_height, img_width, CV_32FC1, Scalar(122.67891434));
    pixes_mat.push_back(B_Mat);
    pixes_mat.push_back(G_Mat);
    pixes_mat.push_back(R_Mat); 
    cv::Mat mean_pixes_mat;
    cv::merge(pixes_mat, mean_pixes_mat); 
    return mean_pixes_mat; 
}

// Function: Assign single img (BGR format) data to mutable cup data
// Note: i bigin from 0 
void assign_img_to_mutable_cpu_data(float* input_data, const cv::Mat &sample_float, const cv::Mat &mean_pixes_mat, int ith){
    cv::Mat img_substract_mean = sample_float - mean_pixes_mat;
    // Splitting each channel is essential, since we must assign all the B channel data to
    // mutable cpu data firstly. We can't use:
    // float* BGR_array = (float*)img_substract_mean.data
    // https://tolleybot.wordpress.com/2011/05/14/accessing-an-opencv-mat-data-using-c/
    std::vector<cv::Mat> input_channels(3);
    split(img_substract_mean, input_channels);
    float* B_array = (float*)input_channels[0].data;
    float* G_array = (float*)input_channels[1].data;
    float* R_array = (float*)input_channels[2].data;

    int Length = img_substract_mean.rows*img_substract_mean.cols;

    // Assign image data to the mutable cpu data
    memcpy( input_data + (3*ith+0)*Length, B_array, sizeof(B_array[0])*Length );
    memcpy( input_data + (3*ith+1)*Length, G_array, sizeof(B_array[0])*Length );
    memcpy( input_data + (3*ith+2)*Length, R_array, sizeof(B_array[0])*Length );
    printf("input number channels: %d\n", input_channels.size());
}

// Function: Transform pool5 feature maps to Cross-dimensional Weighting feature
std::vector<float> extract_CDW_feat(float * feature_blob_data, int map_height, int map_width, int blob_num_channels){
    // Compute spatial weight
    std::vector<MatrixXf> maps;  // to do: sparse matrix
    MatrixXf sum_maps = MatrixXf::Zero(map_height, map_width);
    array_to_maps(feature_blob_data, maps, sum_maps, map_height, map_width, blob_num_channels);
    MatrixXf spatial_weight = compute_spatial_weight(sum_maps);

    // Compute channel weight
    MatrixXf channel_weight = MatrixXf::Zero(1, blob_num_channels);
    compute_channel_weight(channel_weight, maps, map_height, map_width, blob_num_channels);

    // Computer crow feature
    MatrixXf feat = MatrixXf::Zero(1, blob_num_channels);
    for (int i = 0; i < blob_num_channels; i++){
        feat(0, i) = maps[i].cwiseProduct(spatial_weight).sum();
    }
    feat = feat.cwiseProduct(channel_weight);

    // feat saved in MatrixXf format transforms into std::vector
    float * tmp_feat = new float[blob_num_channels];
    Map<MatrixXf>( tmp_feat, feat.rows(), feat.cols() ) = feat;
    std::vector<float> feature(tmp_feat, tmp_feat + blob_num_channels);
    delete [] tmp_feat;
    return feature;
}

int main(int argc, char** argv) { 

    // Here we use GPU only mode
    Caffe::SetDevice(6);
    Caffe::set_mode(Caffe::GPU);

    // Read multi-images
    std::vector<cv::Mat> imgs;

    std::string image_path("/home/yuanyong/test12.jpg");
    cv::Mat img = cv::imread(image_path, -1);
    imgs.push_back(img);

    imgs.push_back(img); // two images
    int num_imgs = imgs.size();
    vector<int> labels(imgs.size(), 0);
    int img_width = img.size().width;
    int img_height = img.size().height;

    printf("image size: H: %d, W: %d\n", img_height, img_width);

    // Net creation
    std::string pretrained_binary_proto("/home/yuanyong/models/VGG_ILSVRC_16_layers.caffemodel");
    std::string feature_extraction_proto("/home/yuanyong/python/crow/vgg/VGG_ILSVRC_16_pool5.prototxt");
    boost::shared_ptr<Net<float> > feature_extraction_net(new Net<float>(feature_extraction_proto, caffe::TEST));
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);
    
    // Input layer
    caffe::Blob<float>* input_layer = feature_extraction_net->input_blobs()[0];

    // Input layer parameters
    int batch_size = input_layer->num();
    int num_channels = input_layer->channels();
    printf("layer input, batch_size: %d, channels: %d, height: %d, width: %d\n", batch_size, num_channels, input_layer->height(), input_layer->width());

    // Reshape input layer
    int new_batch_size = 2;
    input_layer->Reshape(new_batch_size, num_channels, img_height, img_width);
    printf("new layer input, batch_size: %d, channels: %d, height: %d, width: %d\n", input_layer->num(), input_layer->channels(), input_layer->height(), input_layer->width());

    // Forward dimension change to all layers
    feature_extraction_net->Reshape();
    
    // Convert the input image to the input image format of the network
    // BGR channel
    cv::Mat sample;
    if (img.channels() == 3 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    // Resizing
    cv::Mat sample_resized;
    sample_resized = sample;
    
    // Float conversion
    cv::Mat sample_float;
    if (num_channels == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
    
    cv::Mat sample_normalized;
    
    // Mean Substraction, per channel
    int width = input_layer->width();
    int height = input_layer->height();
    cv::Mat mean_pixes_mat = mean_mat_generation(height, width);
    
    // Assign image data to mutable cpu data
    float* input_data = input_layer->mutable_cpu_data();
    assign_img_to_mutable_cpu_data(input_data, sample_float, mean_pixes_mat, 0);
    assign_img_to_mutable_cpu_data(input_data, sample_float, mean_pixes_mat, 1);

    // Actual feature computation
    feature_extraction_net->Forward();
    
    // Select pool5 as the feature 
    const boost::shared_ptr<Blob<float> > feature_blob = feature_extraction_net->blob_by_name("pool5");
    
    // Obtain the feature
    const float* feature_blob_data;
    feature_blob_data = feature_blob->cpu_data();
    
    int dim_feats = feature_blob->count() / batch_size;
    int map_height = feature_blob->height();
    int map_width = feature_blob->width();
    int blob_num_channels = feature_blob->channels();
 
    printf("channels: %d, width: %d, hight: %d, batch_size: %d, dim_features: %d\n", feature_blob->channels(), feature_blob->width(), feature_blob->height(), feature_blob->num(), dim_feats);

    int dim_per_feat = dim_feats / feature_blob->num();
    
    printf("dim_per_feat: %d\n", dim_per_feat);     

    float * tmp_blob_data = new float[dim_per_feat];

    memcpy(tmp_blob_data, feature_blob_data + 0*dim_per_feat, sizeof(feature_blob_data[0])*dim_per_feat);
    std::vector<float> feat0 = extract_CDW_feat(tmp_blob_data, map_height, map_width, blob_num_channels);

    memcpy(tmp_blob_data, feature_blob_data + 1*dim_per_feat, sizeof(feature_blob_data[0])*dim_per_feat);
    std::vector<float> feat1 = extract_CDW_feat(tmp_blob_data, map_height, map_width, blob_num_channels);
    
    for(vector<float>::iterator it = feat1.begin(); it != feat1.end(); ++it){
        printf("%f ", *it);
    }
    
    delete [] tmp_blob_data;

    return 1; 
}
