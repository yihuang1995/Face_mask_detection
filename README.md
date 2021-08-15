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
<p align="center">
  <img src="https://github.com/yihuang1995/Face_mask_detection/blob/main/Images/pipeline.png">
</p>

## YOLOv5 Face & Mask Detection:
Detection instructions are in 'yolov5_face_detection.ipynb', download YOLOv5 from github <br>
Run python detect.py --source 0 --weights 'best.pt' in the YOLOv5 repo 

## Data Source:
https://www.kaggle.com/andrewmvd/face-mask-detection <br>
https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset <br>
https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning <br>

## Experiments on Dataset
When training MobileNetv2 mask classifier, we tested performance of the model on three different training dataset. The first one is trained on only the fake mask dataset, the second one is trained on images with real mask, and the last one is two dataset combined.
<p align="center">
  <img src="https://github.com/yihuang1995/Face_mask_detection/blob/main/Images/dataset_comparison.png">
</p>

The validation dataset for this comparison is two dataset combiend and we balanced the images from these dataset, and we trained each model in 10 epoches. The fake mask dataset's performance is somehow better than the real one. It might due to the reason that the real mask dataset has low resolution and also the images only contain the faces without the background, it losses information other than the faces. The two dataset combined has the highest accuracy on the validation dataset.

## Applications:
**Single Face:**<br>
<p align="center">
  <img src="https://github.com/yihuang1995/Face_mask_detection/blob/main/Images/yann_detect.png">
</p>

**Multiple Faces:**<br>
<p align="center">
  <img src="https://github.com/yihuang1995/Face_mask_detection/blob/main/Images/multi_detection.png">
</p>
The photo is detected by FaceNet + MobileNetv2 combination model. It gives the right prediction for all faces.
<br/><br/>

<p align="center">
  <img src="https://github.com/yihuang1995/Face_mask_detection/blob/main/Images/multi_detect2.png">
</p>
This photo is also detected by FaceNet + MobileNetv2 combination model. For two left faces, the prediction is wrong, we believe it is due to the limitation of the training data.
<br/><br/>

<p align="center">
  <img src="https://github.com/yihuang1995/Face_mask_detection/blob/main/Images/yolov5_detect.png">
</p>
This photo is detected by YOLOv5 model trained on images with bounding boxes and labels. The performance is better than the FaceNet + MobileNetv2 combo.
<br/><br/>

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
