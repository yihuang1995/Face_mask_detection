# Face & Mask Detection
Created by Yi Huang & Ricky Zhang

## FaceNet + MobileNetv2:
**Key Files:**  <br>
Mask_Classifier.ipynb     -> notebook used to train mask detector <br>
face_detect.ipynb         -> apply face detector and mask detector on images <br>
video_mask_detect_yi.py   -> python app that deploys mask detector with real-time video. Run with **python video_mask_detect_yi.py** <br>
<br>
**Other Files:**  <br>
Dataset Comparison.ipynb  -> Comparing performance of models trained on different dataset  <br>
MobileNetV2.py            -> implement MobileNet V2 architecture with PyTorch <br>
MobileNetV2_mask_fulldata -> Final Model  <br>

## Pipeline:
![image](https://github.com/yihuang1995/Face_mask_detection/blob/main/Images/pipeline.png)

## YOLOv5 Face & Mask Detection:
Detection instructions are in 'yolov5_face_detection.ipynb', download YOLOv5 from github <br>
Run python detect.py --source 0 --weights 'best.pt' in the YOLOv5 repo 

## Data Source:
https://www.kaggle.com/andrewmvd/face-mask-detection <br>
https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset <br>
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning <br>

## Applications:
**Image:**<br>
![image](https://user-images.githubusercontent.com/72787973/129308512-31e78558-fdd7-4ab3-bc1c-a248ae15415a.png)
![image](https://user-images.githubusercontent.com/72787973/129308523-325dc78b-36aa-4d8a-9603-474e763bcd98.png)
![image](https://user-images.githubusercontent.com/72787973/129308529-60b8f411-35ec-421b-9d79-b4a1d74024e6.png)

**Video:**
https://youtu.be/jXvMSskHRbo

## Things Need to Improve:
**Face Detection is not stable under poor light condition:**<br>
https://youtu.be/b_ydhvaXcG4

**Wrong classification when mouth is coverd:**<br>
https://youtu.be/YopM2upd0iQ

**High latency for the YOLOv5 model:**<br>
https://youtu.be/Mr98Dw70jJk

## Conclusion:
YOLOv5's performance is better than FaceNet+MobileNetv2 combined. Using image with bounding box and labels can obtrain a better model, whereas this type of dataset is often much less than image with label dataset. The whole bottleneck of face and mask detection is the limitation of dataset, images under different conditions can be added in the future to train a more powerful mask detection model.
# Face_mask_detection
