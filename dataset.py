import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
import json
import dlib
import shutil
from PIL import Image, ImageFilter

temp_dir = 'data/temp'
orig_dir = 'data/images'
train_dir = 'data/train'
val_dir = 'data/val'
drivFace_images = "C:/Users/selloh/Desktop/Datasets/DrivFace/DrivImages"
# drivFace_images = './datasets/Drivface/DrivImages'
# drivFace_annotations = './datasets/DrivFace/drivPoints.txt'
drivFace_annotations = 'C:/Users/selloh/Desktop/Datasets/DrivFace/drivPoints.txt'

hdf5_path = 'DrivFace.h5'

# face_detector = dlib.get_frontal_face_detector()
# landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def load_dataset():
     returnNone

def load_annotations(image_path):
    base_name = os.path.basename(image_path).split('.')[0]
    annotations = pd.read_csv(drivFace_annotations)
    annotation = annotations[annotations['fileName'].str.contains(base_name)].iloc[0]

    landmarks = {
        "xF": annotation['xF'], "yF": annotation['yF'], "wF": annotation['wF'], "hF": annotation['hF'],
        "xRE": annotation['xRE'], "yRE": annotation['yRE'],
        "xLE": annotation['xLE'], "yLE": annotation['yLE'],
        "xN": annotation['xN'], "yN": annotation['yN'],
        "xRM": annotation['xRM'], "yRM": annotation['yRM'],
        "xLM": annotation['xLM'], "yLM": annotation['yLM']
    }

    pose = annotation['label']

    return landmarks, pose

def split_dataset(orig_dir, train_dir, val_dir, split_ratio):

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        print('Creating train directories for each class')
        train_ff_dir = os.path.join(train_dir, 'ff')
        os.makedirs(train_ff_dir)
        train_ll_dir = os.path.join(train_dir, 'll')
        os.makedirs(train_ll_dir)
        train_lr_dir = os.path.join(train_dir, 'lr')
        os.makedirs(train_lr_dir)

        print("Creating val directories for each class")
        val_ff_dir = os.path.join(val_dir, 'ff')
        os.makedirs(val_ff_dir)
        val_ll_dir = os.path.join(val_dir, 'll')
        os.makedirs(val_ll_dir)
        val_lr_dir = os.path.join(val_dir, 'lr')
        os.makedirs(val_lr_dir)

        files = pd.read_csv(drivFace_annotations)['fileName'].tolist()

        files_ff = [file for file in files if 'f' in file]
        files_ll = [file for file in files if 'll' in file]
        files_lr = [file for file in files if 'lr' in file]

        i = 0
        for fname in files_ff:
            src = os.path.join(orig_dir, fname + '.jpg')
            dst_train = os.path.join(train_ff_dir, fname + '.jpg')
            dst_val = os.path.join(val_ff_dir, fname + '.jpg')
            i += 1
            if i < len(files_ff)*split_ratio:
                shutil.move(src, dst_train)
            else:
                shutil.move(src, dst_val)

            
        i = 0 
        for fname in files_ll:
            src = os.path.join(orig_dir, fname + '.jpg')
            dst_train = os.path.join(val_ll_dir, fname + '.jpg')

            i += 1
            if i < len(files_ll)*split_ratio:
                shutil.move(src, dst)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224,224))
    image = np.array(image)
    image = image / 255.0
    
    # image = cv2.imread(image)
    # image = image / 255.0 # normalze to [0,1]

    landmarks, pose = load_annotations(image_path)

    return image, landmarks, pose

image_paths = glob(os.path.join(drivFace_images, '*.jpg'))

for image_path in image_paths:
    image, landmarks, pose = preprocess_image(image_path)
    print(landmarks)

