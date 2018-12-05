import os
import numpy as np
import cv2
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "src/images")

face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):  #printing path of file
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()  # getting folder name
            # print(label, path)
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            #x_train.append(path)
            #y_labels.append(label)
            pil_image = Image.open(path).convert("L") # convert image to grayscale

            # resizing
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            # image to array
            image_array = np.array(final_image, "uint8")  #uint8 is a type
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


#print(y_labels)
#print(x_train)

with open("labels.pkl", 'wb') as f: #wb ,writing byte
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")


