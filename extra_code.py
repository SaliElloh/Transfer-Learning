# def get_list_from_filenames(file_path):
#     # input:    relative path to .txt file with file names
#     # output:   list of relative path names
#     with open(file_path) as f:
#         lines = f.read().splitlines()
#     return lines


# #preprocessing the Pose 300W LP dataset

# #head movement
# class Pose_300W_LP(Dataset):
#     def __init__(self, data_dir, filename_path, transform, img_ext='.jpg',annot_ext='.mat', image_mode='RGB'):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.img_ext = img_ext
#         self.annot_ext =  annot_ext


#         filename_list = get_list_from_filenames(filename_path)

#         self.X_train = filename_list
#         self.y_train = filename_list
#         self.image_mode = image_mode
#         self.length = len(filename_list)


#     def __getitem__(self, index):
#         img = Image.open(os.path.join(self.data_dir, self.X_train[index] + '_rgb' + self.img_ext))
#         img = img.convert(self.image_mode)
#         pose_path = os.path.join(self.data_dir, self.y_train[index] + '_pose' + self.annot_ext)

#         y_train_list = self.y_train[index].split('/')
#         bbox_path = os.path.join(self.data_dir, y_train_list[0] + '/dockerface-' + y_train_list[-1] + '_rgb' + self.annot_ext)

#         # Load bounding box
#         bbox = open(bbox_path, 'r')
#         line = bbox.readline().split(' ')
#         if len(line) < 4:
#             x_min, y_min, x_max, y_max = 0, 0, img.size[0], img.size[1]
#         else:
#             x_min, y_min, x_max, y_max = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
#         bbox.close()

#         # Load pose in degrees
#         pose_annot = open(pose_path, 'r')
#         R = []
#         for line in pose_annot:
#             line = line.strip('\n').split(' ')
#             l = []
#             if line[0] != '':
#                 for nb in line:
#                     if nb == '':
#                         continue
#                     l.append(float(nb))
#                 R.append(l)

#         R = np.array(R)
#         T = R[3,:]
#         R = R[:3,:]
#         pose_annot.close()

#         R = np.transpose(R)

#         roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
#         yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
#         pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

#         # Loosely crop face
#         k = 0.35
#         x_min -= 0.6 * k * abs(x_max - x_min)
#         y_min -= k * abs(y_max - y_min)
#         x_max += 0.6 * k * abs(x_max - x_min)
#         y_max += 0.6 * k * abs(y_max - y_min)
#         img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

#         # Bin values
#         bins = np.array(range(-99, 102, 3))
#         binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

#         labels = torch.LongTensor(binned_pose)
#         cont_labels = torch.FloatTensor([yaw, pitch, roll])

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, labels, cont_labels, self.X_train[index]

#     def __len__(self):
#         # 15,667
#         return self.length
    

#Preprocessing the DrivFace Dataset
# Shape of input data
HEIGHT = 224
WIDTH = 224
CHANNELS = 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)



# def load_dataset():
#   if not os.path.exists(train_directory):
#     os.makedirs(temp_directory)
#     os.makedirs(train_directory)
#     print("Downloading dataset to "+ temp_directory)
#     file = wget.download(url, out=temp_directory)
#     print("\nUnzipping the files..")
#     pzf = PyZipFile(file)
#     pzf.extractall(temp_directory)
#     pzf = PyZipFile(temp_directory+'/DrivFace/DrivImages.zip')
#     pzf.extractall(temp_directory)
#     print("Moving files to "+train_directory)
#     for file in os.listdir(temp_directory+'/DrivImages'):
#       shutil.move(temp_directory+'/DrivImages/'+file,train_directory+'/'+file)
#     shutil.move(temp_directory+'/DrivFace/drivPoints.txt',train_directory+'/drivPoints.txt')
#     print("Deleting temporary directory "+ temp_directory)
#     shutil.rmtree(temp_directory)

# def create_augumented_images(directory):
#     datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=False,
#         fill_mode='nearest')
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     else:
#         return
#     print("Generating augumented images in the directory " + directory)
#     for index, row in df_label.iterrows():
#             file = row['fileName'] + ".jpg"
#             image = cv2.imread(train_directory + '/' + file)
#             y = int(row['yF'])
#             x = int(row['xF'])
#             w = int(row['wF'])
#             h = int(row['hF'])
#             #image = image[y:y+h, x:x+w]
#             image = cv2.resize(image, (HEIGHT, WIDTH), interpolation = cv2.INTER_AREA)
#             X = img_to_array(image)
#             X = X.reshape((1,) + X.shape)
#             i = 0
#             for batch in datagen.flow(X, batch_size=1,
#                               save_to_dir=directory, save_prefix=file, save_format='jpg'):
#                 i += 1
#                 if i > 20:
#                     break  # otherwise the generator would loop indefinitely

# def proc_images():
#     num_images = 0
#     dataset_X = []
#     dataset_y = []
#     if os.path.isfile(hdf5_path):
#         return
#     else:
#         print("writing images to " + hdf5_path)
#         with h5py.File(hdf5_path, mode='w') as hf:
#             for index, row in df_label.iterrows():
#                 file = row['fileName'] + ".jpg"
#                 image = cv2.imread(train_directory + '/' + file)
#                 y = int(row['yF'])
#                 x = int(row['xF'])
#                 w = int(row['wF'])
#                 h = int(row['hF'])
#                 #image = image[y:y+h, x:x+w]
#                 image = cv2.resize(image, (HEIGHT, WIDTH), interpolation = cv2.INTER_AREA)
#                 dataset_X.append(image)
#                 dataset_y.append(row['label'])
#                 num_images = num_images + 1
#                 for augumented_image in glob.glob(train_directory + '/augumented_images/' + file + "*"):
#                     image = cv2.imread(augumented_image)
#                     y = int(row['yF'])
#                     x = int(row['xF'])
#                     w = int(row['wF'])
#                     h = int(row['hF'])
#                     #image = image[y:y+h, x:x+w]
#                     image = cv2.resize(image, (HEIGHT, WIDTH), interpolation = cv2.INTER_AREA)
#                     dataset_X.append(image)
#                     dataset_y.append(row['label'])
#                     num_images = num_images + 1
#             dataset_X = hf.create_dataset(
#                     name='dataset_X',
#                     data=dataset_X,
#                     shape=(len(dataset_X), HEIGHT, WIDTH, CHANNELS),
#                     maxshape=(len(dataset_X), HEIGHT, WIDTH, CHANNELS),
#                     compression="gzip",
#                     compression_opts=9)
#             dataset_y = hf.create_dataset(
#                     name='dataset_y',
#                     data=dataset_y,
#                     shape=(len(dataset_y), 1,),
#                     maxshape=(len(dataset_y), None,),
#                     compression="gzip",
#                     compression_opts=9)
#             number_images = next(os.walk(train_directory))[2]
#             number_images_aug = next(os.walk(aug_images_directory))[2]
#             print("Number of images written to "+ hdf5_path + " is: " + str(len(number_images)+len(number_images_aug)))

# load_dataset()
# df_label = pd.read_csv(train_directory+'/drivPoints.txt')
# df_label = df_label.dropna()
# create_augumented_images(aug_images_directory)
# proc_images()



#Preprocessing the FER-2013 Dataset




#Preprocessing the KMU-FED Dataset



