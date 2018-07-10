'''
MAKE A PREDICTION

A FACE IS A VALID PREDICTION IF THE CLOSEST NEIGHBOR IN THE TRAINING SET HAS A EUCLIDEAN DISTANCE OF LESS THAN 'distance_threshold' (default = 0.6)


---- Below text from https://github.com/ageitgey/face_recognition -----
Deployment to Cloud Hosts (Heroku, AWS, etc)
Since face_recognition depends on dlib which is written in C++,
it can be tricky to deploy an app using it to a cloud hosting provider like Heroku or AWS.

To make things easier, there's an example Dockerfile in this repo that
shows how to run an app built with face_recognition in a Docker container.
With that, you should be able to deploy to any service that supports Docker images.
'''

import face_recognition
from PIL import Image, ImageDraw
from sklearn.externals import joblib
import numpy as np
from face_recognition.face_recognition_cli import image_files_in_folder
import os

knn_clf = joblib.load('knn.sav')
distance_threshold = 0.5

dir = 'Intern01'

for img_path in image_files_in_folder(dir):
    print(img_path)

    filename = os.path.basename(img_path)
    image = face_recognition.load_image_file(img_path)
    face_bounding_boxes = face_recognition.face_locations(image)

    pil_image = Image.fromarray(image)
    has_zteam_member = False
    if len(face_bounding_boxes) > 0:
        # print('Found ' + str(len(face_bounding_boxes)) + ' faces!')
        face_encodings = face_recognition.face_encodings(face_image=image, known_face_locations=face_bounding_boxes)
        face_predictions = knn_clf.predict(face_encodings)
        closest_distances, indexes = knn_clf.kneighbors(face_encodings, n_neighbors=1)

        for i, (face_prediction, closest_distance) in enumerate(zip(face_predictions, closest_distances)):

            # if there's a zteam member, set the has zteam member flag to True
            # so we don't save the image in the Other folder
            # print('Face #' + str(i) + ':  Closest distance = ' + str(closest_distance))
            if closest_distance <= distance_threshold:
                # print('Zteam member found!')
                has_zteam_member = True
                path = os.path.join(dir, face_prediction)

                if os.path.exists(path) == False:
                    os.mkdir(path)

                pil_image.save(os.path.join(path, filename))
                # print('Saving to ' + path)
            # if it's the last face and we haven't found a zteam member yet,
            # save the image to the Other folder
            elif i == len(face_predictions) - 1 and has_zteam_member == False:
                continue
                # print('No zteam member found')
                # path = os.path.join(dir, 'Other')
                # if os.path.exists(path) == False:
                #     os.mkdir(path)
                # pil_image.save(os.path.join(path, filename))
                # print('Saving to ' + path)
    else:
        continue
        # print('No faces found!')
        # path = os.path.join(dir, 'Other')
        # if os.path.exists(path) == False:
        #     os.mkdir(path)
        # pil_image.save(os.path.join(path, filename))
        # print('Saving to ' + path)
