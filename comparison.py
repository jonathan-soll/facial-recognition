import face_recognition
from PIL import Image, ImageDraw

image1 = face_recognition.load_image_file('predict/Multiple/you_and_krause.jpeg')
image1_face_locations = face_recognition.face_locations(image1)

image2 = face_recognition.load_image_file('train\\Dan_Gilbert\\gty_691347576.jpg')
image2_face_locations = face_recognition.face_locations(image2)

pil_image1 = Image.fromarray(image1)
# draw1 = ImageDraw.Draw(pil_image1)
#
pil_image2 = Image.fromarray(image2)
# draw2 = ImageDraw.Draw(pil_image2)

face_location1 = image1_face_locations[len(image1_face_locations)-1]
(top1, right1, bottom1, left1) = face_location1
# draw1.rectangle(((left1, top1), (right1, bottom1)), outline=(0, 0, 0))
# pil_image1.show()
cropped_image1 = pil_image1.crop((left1, top1, right1, bottom1))
cropped_image1.show()

face_location2 = image2_face_locations[len(image2_face_locations)-1]
(top2, right2, bottom2, left2) = face_location2
# draw2.rectangle(((left2, top2), (right2, bottom2)), outline=(0, 0, 0))
# pil_image2.show()
cropped_image2 = pil_image2.crop((left2, top2, right2, bottom2))
cropped_image2.show()

# print(face_location1)
# print(face_location2)

image1_encoding = face_recognition.face_encodings(image1, [face_location1])
image2_encoding = face_recognition.face_encodings(image2, [face_location2])

# print(image1_encoding)
# print(image2_encoding)

face_distance = face_recognition.face_distance(image1_encoding, image2_encoding[0])
print('Distance  = ' + str(face_distance))
