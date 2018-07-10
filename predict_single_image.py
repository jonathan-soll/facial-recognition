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

knn_clf = joblib.load('knn.sav')
distance_threshold = 0.5

# new_image_path = 'predict/Dave_Schroeder/Dave_Schroeder (13).jpg'
# new_image_path = 'predict/Tim_Birkmeier/Tim_Birkmeier (4).jpg'
# new_image_path = 'predict/Heather_Lovier/Heather_Lovier (10).jpg'
# new_image_path = 'predict/Mike_Malloy/Mike_Malloy (11).jpg'
# new_image_path = 'predict/None/bill_emerson.jpg'
# new_image_path = 'predict/None/guy01.jpeg'
# new_image_path = 'predict/Tim_Birkmeier/tim01.jpeg'
# new_image_path = 'predict/Multiple/you_and_krause.jpeg'
# new_image_path = 'predict/Multiple/testing2.jpg'
# new_image_path = 'predict/Tim_Birkmeier/tim02.jpeg'
# new_image_path = 'predict/Linglong_He/Linglong001.jpeg'
# new_image_path = 'predict/None/bt.jpg'
# new_image_path = 'predict/None/js.jpg'
new_image_path = 'Intern01/InternClosingEvent-20161201-041.jpg'

new_image = face_recognition.load_image_file(new_image_path)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(new_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)


face_locations = face_recognition.face_locations(new_image)
if len(face_locations) == 0:
    print('No faces!')
else:
    face_encodings = face_recognition.face_encodings(face_image=new_image, known_face_locations=face_locations)

    # Use the KNN model to find the best matches for the test face
    # closest_distances:  each element is the closest n distances for each face
    # closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=5)
    closest_distances, indexes = knn_clf.kneighbors(face_encodings, n_neighbors=5)

    # iterate through each face and create a list
    # are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]
    are_matches = [closest_distances[i][0] <= distance_threshold for i in range(len(face_locations))]

    face_predictions = knn_clf.predict(face_encodings)

    # print(closest_distances)
    # print(are_matches)
    # print(face_predictions)
    # print(indexes)

    training_imagepaths = np.load('training_imagepaths.npy')
    training_names = np.load('training_names.npy')
    # print(training_imagepaths[indexes])
    # print(training_names[indexes])


    for pred, loc, rec, closest_dist in zip(face_predictions, face_locations, are_matches, closest_distances):
        # print(closest_dist)
        (top, right, bottom, left) = loc
        col = (255, 0, 0)

        name = pred.encode("UTF-8")

        if rec == True:
            col = (0, 255, 0)

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=col)
        draw.text((left + 6, bottom + 20), str(np.round(closest_dist, 2)), fill=(0, 255, 255, 255))
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.text((left + 6, bottom - text_height + 10), name, fill=(0, 255, 255, 255))


        name = pred.encode("UTF-8")

pil_image.show()
