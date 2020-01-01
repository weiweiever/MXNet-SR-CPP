#include <iostream>
#include <opencv2/opencv.hpp>
#include "mxnet/c_predict_api.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace cv;

//int main() {
//    VideoCapture cap(0);
//    Mat frame;
//
//    while(1){
//        cap>>frame;
//        cout<<frame.rows<< "  "<<frame.cols<<endl;
//        imshow("frame",frame);
//        if(waitKey(1)=='q')
//            break;
//    }
//    return 0;
//}