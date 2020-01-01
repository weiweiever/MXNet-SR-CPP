
#include "predict.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <iomanip>
#include <opencv2/opencv.hpp>
// Path for c_predict_api
#include <mxnet/c_predict_api.h>

using namespace std;

const mx_float DEFAULT_MEAN = 117.5;

static std::string trim(const std::string& input) {
    auto not_space = [](int ch) {
        return !std::isspace(ch);
    };
    auto output = input;
    output.erase(output.begin(), std::find_if(output.begin(), output.end(), not_space));
    output.erase(std::find_if(output.rbegin(), output.rend(), not_space).base(), output.end());
    return output;
}

// Read file to buffer
class BufferFile {
public :
    std::string file_path_;
    std::size_t length_ = 0;
    std::unique_ptr<char[]> buffer_;

    explicit BufferFile(const std::string& file_path)
            : file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = static_cast<std::size_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

        // Buffer as null terminated to be converted to string
        buffer_.reset(new char[length_ + 1]);
        buffer_[length_] = 0;
        ifs.read(buffer_.get(), length_);
        ifs.close();
    }

    std::size_t GetLength() {
        return length_;
    }

    char* GetBuffer() {
        return buffer_.get();
    }
};

void GetImageFile(const std::string& image_file,
                  mx_float* image_data, int channels,
                  cv::Size resize_size, const mx_float* mean_data = nullptr) {


}

void predict(PredictorHandle pred_hnd, const std::vector<mx_float> &image_data) {


}

int main(int argc, char* argv[]) {

    std::string test_file("/home/zhiyu/Projects/facedetect/SR/myImage/qq.jpeg");

    // Models path for your model, you have to modify it
    std::string json_file = "../models/WDSR-a-8-x2-symbol.json";
    std::string param_file = "../models/WDSR-a-8-x2-0000.params";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    // Parameters
    int dev_type = 2;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = { "data" };
    const char** input_keys = input_key;

    // Image size and channels
    int width = 210;
    int height = 240;
    int channels = 1;

    const mx_uint input_shape_indptr[2] = { 0, 4 };
    const mx_uint input_shape_data[4] = { 1,
                                          static_cast<mx_uint>(channels),
                                          static_cast<mx_uint>(height),
                                          static_cast<mx_uint>(width) };

    if (json_data.GetLength() == 0 || param_data.GetLength() == 0) {
        return EXIT_FAILURE;
    }



    // Read Image Data
    std::vector<float> image_data(width * height);
    std::vector<cv::Mat> img_chs;
    //GetImageFile(test_file, image_data.data(), channels, cv::Size(width, height), nd_data);
    cv::Mat im_ori = cv::imread(test_file, cv::IMREAD_COLOR);
    im_ori.convertTo(im_ori,CV_32FC3);
    cv::cvtColor(im_ori,im_ori,cv::COLOR_BGR2YCrCb);
    img_chs.push_back(cv::Mat(width,height,CV_32FC1,image_data.data()));
    img_chs.push_back(cv::Mat(width,height,CV_32FC1));
    img_chs.push_back(cv::Mat(width,height,CV_32FC1));

    cv::split(im_ori,img_chs);




    // Create Predictor
    PredictorHandle pred_hnd;
    MXPredCreate(static_cast<const char*>(json_data.GetBuffer()),
                 static_cast<const char*>(param_data.GetBuffer()),
                 static_cast<int>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 &pred_hnd);
    assert(pred_hnd);

    MXPredSetInput(pred_hnd, "data", image_data.data(), image_data.size());
    MXPredForward(pred_hnd);

    mx_uint *shape = NULL;
    mx_uint shape_len = 0;

    MXPredGetOutputShape(pred_hnd,0,&shape,&shape_len);
    int reg_size=1;

    for(unsigned int i=0;i<shape_len;i++)
        reg_size*=shape[i];
    std::vector<mx_float> reg(reg_size);
    MXPredGetOutput(pred_hnd,0,reg.data(),reg_size);

    cv::Mat out(height*2,width*2,CV_32FC1,reg.data());
    //out.convertTo(out,CV_8UC1);

    cv::Mat dst;
    std::vector<cv::Mat> dst_chs;

    cv::resize(img_chs[1],img_chs[1],cv::Size(0,0),2,2,cv::INTER_CUBIC);
    cv::resize(img_chs[2],img_chs[2],cv::Size(0,0),2,2,cv::INTER_CUBIC);

    dst_chs.push_back(out);
    dst_chs.push_back(img_chs[1]);
    dst_chs.push_back(img_chs[2]);
    std::cout<<dst_chs[0].depth()<<"  "<<dst_chs[1].depth()<<" "<<dst_chs[2].depth()<<std::endl;
    cv::merge(dst_chs,dst);
    cv::cvtColor(dst,dst,cv::COLOR_YCrCb2BGR);
    dst.convertTo(dst,CV_8UC3);

    imshow("out",dst);
    cv::waitKey(0);
    // Release Predictor
    MXPredFree(pred_hnd);


    //predict(pred_hnd, image_data);


    return EXIT_SUCCESS;
}