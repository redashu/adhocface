import cv2
import os
import numpy as np
name = input("Enter subject name : ")

try :
    os.mkdir(name)
except :
    print()

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y + w, x:x + h], faces[0]

def prepare_training_data(subject_dir_path):
    faces = []
    labels = []
    label = 0
    subject_images_names = os.listdir(subject_dir_path)

    for image_name in subject_images_names:
        # ignore system files like .DS_Store
        if image_name.startswith("."):
            continue;

        image_path = subject_dir_path + "/" + image_name
        image = cv2.imread(image_path)
        face, rect = detect_face(image)

        if face is not None:
            faces.append(face)
            labels.append(label)

    return faces, labels

print("Preparing data...")

#Take and save images here
camera = cv2.VideoCapture(0)

def get_image():
    retval, im = camera.read()
    return im


for i in range(10):
    camera_capture = get_image()
    file = name+"/"+str(i)+".jpg"
    cv2.imwrite(file, camera_capture)

del (camera)

faces, labels = prepare_training_data(name)
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

facefile = open(name+"faces.txt", "w")
facefile.write(str(faces))
facefile.close()

