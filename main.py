from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
# import RPi.GPIO as IO
# import blinkBuzz as bb
# import BLEthread as bt
import threading


# initialize the frame counters and the total number of eye closes at different stages
COUNTER = 0
BLINK = 0
STAGE1 = 0
STAGE2 = 0
STAGE3 = 0

COUNTER2 = 0
YAWN = 0
YAWN_TIME = 0


# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# mouth indexes
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

while True:
    t = time.time()

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbor=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    #detect faces in the grayscale frae

    #loop over the face detections:

    # for (x, y, w, h ) in reacts
