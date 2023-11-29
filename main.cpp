#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include "yolo.h"
using namespace cv;
using namespace std;

void detectfuction(string modelpath, string classespath, Mat& img)
{
	my_Configuration config;
	config.confThreshold = 0.5;
	config.modelpath = modelpath;
	config.nmsThreshold = 0.3;
	config.objThreshold = 0.5;

	YOLO my_yolo(config, false, classespath);

	vector<Output> result;
	my_yolo.detect(img, result);
	my_yolo.drawPred(img, result, my_yolo.color);
}
int main(int argc, char** agrv)
{
	Mat img = imread("C:/Users/tianmingYun/Desktop/bus.jpg");
	if (img.empty())
	{
		cout << "no image" << endl;
		return -1;
	}
	string model_path = "./yolov7-tiny.onnx";  //onnx模型路径
	string classespath = "./coco.txt";  //coco类别路径
	detectfuction(model_path, classespath, img);
	

	return 0;
}
