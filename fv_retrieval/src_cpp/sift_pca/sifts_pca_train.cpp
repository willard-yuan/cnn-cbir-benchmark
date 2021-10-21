
#include <iostream>
#include <fstream>
#include "base_utils.hpp"
#include "opencv_utils.hpp"
#include "BinaryCvMat.h"
#include "pca_utils.h"

#define MAX_SIFT_NUM 256000

const int dim_sift = 128;
const int num_reduced_dim = 32;


// shuttle opencv mat
cv::Mat shuffleRows(const cv::Mat &matrix)
{
    std::vector <int> seeds;
    for (int cont = 0; cont < matrix.rows; cont++)
        seeds.push_back(cont);
    
    // shuttle opencv Mat
    cv::randShuffle(seeds);
    
    cv::Mat output;
    if (matrix.rows >= 51){
        for (int cont = 0; cont < 51; cont++){
            output.push_back(matrix.row(seeds[cont]));
        }
    }else{
        output = matrix;
    }
    
    return output;
}

// 选取用于训练的SIFT特征: 方案1
void selectSifts1(const std::string bsifts_path, cv::Mat & train_sifts)
{
    std::ifstream myfile(bsifts_path);
    std::string line;
    int count = 0;
    int j = 0;
    while (std::getline(myfile, line))
    {
        // 载入sift特征
        cv::Mat tmp;
        if(LoadMatBinary(line, tmp)){
            //printf("%d, %s\n", j, line.c_str());
            if(tmp.rows > 0){
                cv::Mat select_sifts = shuffleRows(tmp);
                for(int i = 0; i < select_sifts.rows; i++){
                    if(count + i < MAX_SIFT_NUM){
                        select_sifts.row(i).copyTo(train_sifts.row(count + i));
                    }else{
                        printf("training sifts are engough ...\n\n");
                        return void();
                    }
                }
                ++j;
                count = count + select_sifts.rows;
                // 获取mat数据类型
                //std::string ty = type2str( tmp.type() );
            }
        }
    }
    return void();
}

// 选取用于训练的SIFT特征: 方案2
void selectSifts2(const std::string bsifts_path, cv::Mat & train_sifts)
{
    int max_rows = 1000000000;
    std::ifstream myfile(bsifts_path);
    std::string line;
    int count = 0;
    int j = 0;
    cv::Mat maxMat(max_rows, dim_sift, CV_32FC1);
    while (std::getline(myfile, line))
    {
        // 载入sift特征
        cv::Mat tmp;
        if(LoadMatBinary(line, tmp)){
            //printf("%d, %s\n", j, line.c_str());
            if(tmp.rows > 0){
                for(int i = 0; i < tmp.rows; i++){
                    tmp.row(i).copyTo(maxMat.row(count + i));
                }
            }
            ++j;
            count = count + tmp.rows;
        }
    }
    cv::Mat allSift = maxMat.rowRange(0, count);
    maxMat.release();
    std::cout << allSift.row(count-1) << std::endl;
    printf("all_sifts: %d rows, %d cols, counts: %d\n", allSift.rows, allSift.cols, count);

    std::vector <int> seeds;
    for (int cont = 0; cont < allSift.rows; cont++)
        seeds.push_back(cont);
    cv::RNG rng(1024);
    cv::randShuffle(seeds, 1.0, &rng);

    for (int cont = 0; cont < train_sifts.rows; cont++)
        allSift.row(seeds[cont]).copyTo(train_sifts.row(cont));
}

int main(int argc, char** argv) {
    
    std::string bsifts_path = "/raid/yuanyong/oxford_fv_test/bsift_path.txt";
    // 保存用于训练的SIFT
    std::string train_sift_mat_path = "/raid/yuanyong/oxford_fv_test/model/sift_128_25w_51_sifts.bsift";
    // 保存训练降维的SIFT
    std::string reducedSIFT_path = "/raid/yuanyong/oxford_fv_test/model/sift_64_25w_51_sifts.bsift";
    // 保存PCA模型
    std::string filename = "/raid/yuanyong/oxford_fv_test/model/pca_64.yml";

    // 选取训练用的SIFT    
    cv::Mat train_sifts = cv::Mat::zeros(MAX_SIFT_NUM, dim_sift, CV_32FC1);
    selectSifts2(bsifts_path, train_sifts); 
   
    // root-sift 
    //cv::sqrt(train_sifts, train_sifts);
    SaveMatBinary(train_sift_mat_path, train_sifts);

    // 检查SIFT特征
    printf("check out the 1th desc ...\n");
    std::cout << train_sifts.row(0) << std::endl;
    printf("check out the last desc ...\n");
    std::cout << train_sifts.row(MAX_SIFT_NUM-1) << std::endl; 
    printf("training sifts size: %d rows, %d cols\n", train_sifts.rows, train_sifts.cols);

    // PCA模型训练 
    printf("start training PCA ...\n");
    trainPCA(train_sifts, num_reduced_dim, filename);
    printf("training PCA finished\n");

    // 对SIFT降维
    cvtk::PCAUtils::getInstance()->loadModel(filename);
    cv::Mat reducedData;
    cvtk::PCAUtils::getInstance()->reduceDim(train_sifts, reducedData);
    printf("reduced sifts mat: %d rows, %d cols\n", reducedData.rows, reducedData.cols);   

    // 保存降维结果
    if(SaveMatBinary(reducedSIFT_path, reducedData))
        printf("sifts reduced to 64 dimension and saved successfully\n");
    return 0;
}
