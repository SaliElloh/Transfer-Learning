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
import os
import pandas as pd
import tarfile 

# csv_dir = 'C:\Users\sally\Downloads\hdbd.tar\hdbd_data'
csv_dir = 'c:/Users/selloh/Downloads/hdbd_data'
extracted_tar_files = 'C:/Users/selloh/Desktop/hdbd_data_x'
def process_csv_files(csv_dir):
    dfs = []
    
    for file in os.listdir(csv_dir):
        file_path  = os.path.join(csv_dir, file)
    
        print(file_path)

        with tarfile.open(file_path, 'r') as tar:
            print(f'extracting current file_path {file_path}')
            tar.extractall(extracted_tar_files)
        # if file.endswith('.csv'):
        #     file_path = os.path.join(csv_dir, file)
        #     print(file_path)
        #     # Read the CSV file into a DataFrame
        #     df = pd.read_csv(file_path)
        #     # Append the DataFrame to the list
        #     dfs.append(df)
    
    # # Concatenate all DataFrames into a single DataFrame
    # combined_df = pd.concat(dfs, ignore_index=True)
    
    # return combined_df

# # # Process the CSV files
combined_data = process_csv_files(csv_dir)

# # Display the combined DataFrame
# print(combined_data.head())


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
