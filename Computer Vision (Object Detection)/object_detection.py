import cv2
import os
import numpy as np
import math

classifier = cv2.CascadeClassifier('./dataset/dataset/haarcascade_frontalface_default.xml')

PATH = "./dataset/dataset/images/train"
train_path = os.listdir(PATH)

face_lists = []
class_lists = []

for index, tdir in enumerate(train_path):
    for filename in os.listdir(f"{PATH}/{tdir}"):
        image_path = f"{PATH}/{tdir}/{filename}"
        load_gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Untuk face detection 
        faces = classifier.detectMultiScale(load_gray_image, scaleFactor = 1.2, minNeighbors = 5)
        if(len(faces) == 1):
            x, y, w, h = faces[0]
            face_image = load_gray_image[y : y + h, x : x + w]
            face_lists.append(face_image)
            class_lists.append(index)
        else:
            continue

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_lists, np.array(class_lists))

# # Testing
test_path = "./dataset/dataset/images/test"
for filename in os.listdir(test_path):
    image_path = f"{test_path}/{filename}"
    image = cv2.imread(image_path)
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    faces = classifier.detectMultiScale(image_gray, scaleFactor = 1.2, minNeighbors = 5)
    
    if len(faces) < 1:
        continue
    else:
        for face_rect in faces:
            x, y, w, h = face_rect
            face_image = image_gray[y : y + h, x : x + w]
            result, confidence = face_recognizer.predict(face_image)
            confidence = math.floor(confidence * 100) / 100
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0), 2) # untuk membuat kotak
            image_text = f"{train_path[result]} : {str(confidence)} %"
            cv2.putText(image, image_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 0), 2)

            cv2.imshow("Result", image)
            cv2.waitKey(0) # bakal wait terus sampai usernya memutuskan untuk close