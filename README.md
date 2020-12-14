# Objectdetection_TFLite_Opencv_Coral_board

TFLite V1
> mobilenetv2_ssd를 tflite v1을 이용하는 api들을 이용하여 training하고, convert한 과정이 있음
TFLite V2
> mobilenetv2_ssd를 tflite v2을 이용하는 api들을 이용하여 training하고, convert한 과정이 있음

with TPU
> TPU이용이 가능하도록 quantize된 모델을 이용하는 방법

without TPU
> TPU 이용이 불가능한 본 프로젝트의 모델을 이용하는 방법

# 메뉴얼 모음
+ object detection tutorial
  - https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
  > 이 문서대로 training을 진행하고, graph freeze도 도움을 받음 (직접적인 ssd_graph.py 쓰는 방법은 없음)
 
+ tensorflow 공식홈페이지의 문서
  - https://www.tensorflow.org/lite/models/object_detection/overview?hl=ko
  
+ 깃헙 tf/model/runningonmobile_tf2
  - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md
  
+ 깃헙 android demo 돌리기
  - https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android

+ tflite 자료 블로그
  - https://github.com/TannerGilbert/Tensorflow-Lite-Object-Detection-with-the-Tensorflow-Object-Detection-API/tree/v1

+ object detection 블로그 (tf1)
  - https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model/tree/tf1

+ 코랄 카메라 이용
  - https://m.blog.naver.com/roboholic84/221861998537
