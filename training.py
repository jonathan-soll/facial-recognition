import math
from sklearn import neighbors
import os
import numpy as np
# import os.path
# import pickle
# from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.externals import joblib

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

X = []
y = []
filepaths = []
names = []

train_dir = 'train'

# loop through each person in the training set
for class_dir in os.listdir(train_dir):

    # loop through each training image for current person
    for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
        image = face_recognition.load_image_file(img_path) # loads an image file into a numpy array (lxwx3)
        face_bounding_boxes = face_recognition.face_locations(image) # uses a deep learning model to find faces in the image

        if len(face_bounding_boxes) == 0:
            print('No faces found, image not suitable for training - ' + img_path)
        elif len(face_bounding_boxes) > 1:
            print('More than one face found, image not suitable for training - ' + img_path)
        else:
            # Add the face encoding for current image to the training SET
            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            y.append(class_dir)
            filepaths.append(img_path)
            names.append(class_dir)

# choose the number of neighbors
n_neighbors = int(round(math.sqrt(len(X))))

knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
knn_clf.fit(X, y)

acc_knn = accuracy_score(y, knn_clf.predict(X))
print('KNN training accuracy = ' + str(100*acc_knn) + '%')


names = np.asarray(names)
names_filepath = 'training_names.npy'
np.save(names_filepath, names)

filepaths = np.asarray(filepaths)
filepaths_filepath = 'training_imagepaths.npy'
np.save(filepaths_filepath, filepaths)

model_filename = 'knn.sav'
joblib.dump(knn_clf, model_filename)
