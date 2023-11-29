# opencv_dnn_yolov7
使用opencvdnn模块部署yolov7.
官方yolov7的onnx模型输出3个特征层,与yolov5不同,
yolov5输出4个，第一个为整合好的输出25200*(c+5).
本代码参考https://blog.csdn.net/dongjuexk/article/details/127548541
