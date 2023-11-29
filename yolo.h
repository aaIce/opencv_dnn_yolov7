#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>   
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <time.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;

struct my_Configuration
{
public:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	std::string modelpath;
};

struct Output {
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
};

class YOLO
{
public:
	vector<Scalar> color;
	YOLO(my_Configuration config, bool ifcuda, string classnamespath);  //构造函数
	void detect(Mat& image, vector<Output>& output);  //检测函数
	void drawPred(Mat& img, vector<Output> result, vector<Scalar> color);
private:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	std::vector<std::string> class_names;
	int num_classes;
	int width_;
	int height_;
	Net net;
	int yolo_anthors[3][6] = { {12, 16,  19, 36,  40, 28}, {36, 75,  76, 55,  72, 146}, {142, 110,  192, 243,  459, 401} };
	int netStride[3] = { 8,16,32 };
	void read_classnames(std::string classnamespath, std::vector<std::string>& class_names);
	Mat resize_image(Mat srcimage, int wigth, int height);
	float sigmoid(float x);
};
