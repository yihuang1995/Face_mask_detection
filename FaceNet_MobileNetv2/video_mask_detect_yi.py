import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# data augmentation
import albumentations as A

# pretrained models
import torchvision
from torchvision import models, transforms

from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import torch
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

from imutils.video import VideoStream
import argparse
import imutils

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def faces_classification(face):
    '''
    input a batch of face image fensor, return list of classification results
    '''
    pred_list = []
    prob_list = []
    for item in face:

        face_resized = cv2.resize(np.array(item.permute(1, 2, 0)), (256, 256))

        transformed = transform(image=face_resized.astype(np.uint8))
        transformed = transformed['image']
        softmax = nn.Softmax(dim=1)
        y_pred = maskNet(transformed.unsqueeze(0))
        prob = max(softmax(y_pred)[0])
        y_pred = torch.argmax(y_pred, dim=1)
        pred_list.append(y_pred)
        prob_list.append(prob)
    return pred_list, prob_list


# def detect_and_predict_mask(frame, faceNet, maskNet):
#     (h, w) = frame.shape[:2]
#     face = faceNet(frame)
#     if face == None:
#         return None,None,None
#     pred_list, prob_list = faces_classification(face)
#     boxes, probs, landmarks = faceNet.detect(frame, landmarks=True)
#     return boxes, pred_list, prob_list



faceNet = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6,0.7,0.7], factor=0.709, post_process=False,
    keep_all=True)

maskNet = torch.load('MobileNetV2_mask_fulldata')
maskNet.eval()


transform = A.Compose([
    # training/valid images have same size
    A.PadIfNeeded(min_height=256, min_width=256, p=1),
    A.CenterCrop(width=224, height=224),
    # normalize
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    # convert to a tensor and move color channels
    ToTensorV2()
])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
frame = vs.read()
frame = imutils.resize(frame, width=400)
cv2.imshow("Frame", frame)
key = cv2.waitKey(1) & 0xFF
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (h, w) = frame.shape[:2]
    face = faceNet(frame)
    if face == None:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue
    pred_list, prob_list = faces_classification(face)
    boxes, probs, landmarks = faceNet.detect(frame, landmarks=True)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred, prob) in zip(boxes, pred_list, prob_list):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if pred == 1 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = f"{label}: {prob:.4f}"

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
