#include "yolo.h"

YOLO::YOLO(my_Configuration config, bool ifcuda, string classnamespath)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	this->width_ = 640;
	this->height_ = 640;
	read_classnames(classnamespath, this->class_names);
	this->num_classes = class_names.size();

	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}

	this->net = readNet(config.modelpath);
}

float YOLO::sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

void YOLO::read_classnames(string classnamespath, std::vector<std::string>& class_names)
{
	//std::vector<std::string> class_names;
	ifstream ifs(string(classnamespath).c_str());
	string line;
	while (getline(ifs, line))
	{
		class_names.push_back(line);
	}
}

Mat YOLO::resize_image(Mat srcimage, int wigth, int height)
{

	Mat img;
	if (srcimage.rows != wigth && srcimage.cols != height)
	{
		resize(srcimage, img, Size(wigth, height), INTER_AREA);  //第四个参数为插值方法
	}
	else
	{
		return srcimage;
	}
	return img;
}

void YOLO::detect(Mat& image, vector<Output>& output)
{
	Mat dstimg = resize_image(image, this->width_, this->height_);
	Mat blob = blobFromImage(dstimg, 1/255.0, Size(this->width_, this->height_), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
	//每层都有矩形框 类别索引 置信度
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;
	float ratioh = (float)image.rows / this->height_;  //得到相对于原图的缩放比例
	float ratiow = (float)image.cols / this->width_;
	int net_width = num_classes + 5;  //类别数+5
	for (size_t stride = 0; stride < outs.size(); stride++)//遍历3个特征层
	{	
		float* pdata = (float*)outs[stride].data;
		int grid_x = (int)(width_ / netStride[stride]);//8 16 32 
		int grid_y = (int)(height_ / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++)
		{
			const float anchor_w = yolo_anthors[stride][anchor * 2];  //得到锚框
			const float anchor_h = yolo_anthors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++)
			{
				for (int j = 0; j < grid_x; j++)
				{
					float obj_conf = sigmoid(pdata[4]);  //含有物体的概率
					if (obj_conf > this->objThreshold)
					{
						Mat scores(1, num_classes, CV_32FC1, pdata + 5);
						Point classIdPoint;   //最大值位置
						double max_class_socre;    //获取最大类别对应的置信值
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = sigmoid(max_class_socre);
						if (max_class_socre > this->confThreshold)
						{
							float x = (sigmoid(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];  //x
							float y = (sigmoid(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];   //y
							float w = powf(sigmoid(pdata[2]) * 2.f, 2.f) * anchor_w;   //w
							float h = powf(sigmoid(pdata[3]) * 2.f, 2.f) * anchor_h;  //h
							int left = (int)(x - 0.5 * w) * ratiow + 0.5;
							int top = (int)(y - 0.5 * h) * ratioh + 0.5;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre * obj_conf);
							boxes.push_back(Rect(left, top, int(w * ratiow), int(h * ratioh)));
						}
					}
					pdata += net_width;
				}
			}
		}
	}
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, nms_result);
	for (size_t i = 0; i < nms_result.size(); i++)
	{
		int idx = nms_result[i];
		Output result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}
	/*imshow("image", image);
	waitKey(0);*/
}

void YOLO::drawPred(Mat& img, vector<Output> result, vector<Scalar> color)   // Draw the predicted bounding box
{
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 2);
		string label = format("%.2f", result[i].confidence);
		label = class_names[result[i].id] + ":" + label;

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	imshow("imageshow", img);
	waitKey();
}
