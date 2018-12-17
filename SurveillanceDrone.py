#####################################################
#
#               Surveillance Drone
# By: Woosung Jang, Morgan Burrows, and Tyler Mooney
#
#####################################################
# This program was created to control a Parrot Bebop 2
# quadcopter drone. When active, the drone will wait
# for a face to be in the line of sight. When a face
# is detected the drone will keep the face in the 
# center of the image. If the face is recognized, 
# the drone will take a photo of the face and label
# it with the correct name.

from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI
import threading
import cv2
import time
import pickle
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

isAlive = False

face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pkl", 'rb') as f: #wb ,writing byte
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}


class UserVision:
    def __init__(self, vision,bebop):
        self.index = 0
        self.vision = vision
        self.filename = "test_image.png"
        self.faceid = 999
        self.bebop = bebop

    def save_pictures(self, args):
        img = self.vision.get_latest_valid_picture()

        if img is not None:   # saving latest img
            self.filename = "test_image.png" # % self.index
            cv2.imwrite(self.filename, img)
            
            self.detect_recognize_face()

    def take_photo(self, id):
        img = self.vision.get_latest_valid_picture()

        if img is not None:
            filename = "photo_{}_{}.png".format(id, self.index)
            self.index += 1
            cv2.imwrite(filename, img)

    def detect_recognize_face(self):
        pil_image = Image.open("test_image.png").convert("L") # convert("L") changes img to grayscale
        final_image = pil_image.resize((550,550), Image.ANTIALIAS)
        image_array = np.array(final_image, "uint8")
        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=3) # higher scale facter might increase accuracy
        for (x, y, w, h) in faces:
            print(x,y,w,h)
            print("Detected")
            roi_color = image_array[y:y+h, x:x+w]
            self.adjust_drone_pos(x,y,h)

            #recognizer
            id_, conf = recognizer.predict(roi_color)
            if conf <= 80:  # and conf <= 85:  # 0 is perfect match  200 is max i guess?
                print(id_)
                print(labels[id_])
                if id_ == 0:
                    self.faceid = 0
                elif id_ == 1:
                    self.faceid = 1
                elif id_ == 2:
                    self.faceid = 2
                else:
                    self.faceid = 3
                self.perform_action()

    def perform_action(self):
        #take photo passing name to function
        if self.faceid == 0:
            self.take_photo(labels[0])
        elif self.faceid == 1:
            self.take_photo(labels[1])
        elif self.faceid == 2:
            self.take_photo(labels[2])
        else:
            self.take_photo('Unknown')
        print("Photo Captured")


    def adjust_drone_pos(self, x, y, h):

        if x < 100:
            # Turn CCW
            print("CCW")
            self.bebop.fly_direct(0,0,-80,0,2)

        elif x > 400:
            # Turn CW
            print("CW")
            self.bebop.fly_direct(0,0,80,0,2)
    
        elif y < 100: 
            # increase altitude
            print("ascend")
            self.bebop.fly_direct(0,0,0,20,1)

        elif y > 400:
            # reduce altitude
            print("descend")
            self.bebop.fly_direct(0,0,0,-20,1)

        elif h < 60:
            # move closer
            print("move forward")
            self.bebop.fly_direct(0,10,0,0,1)

        elif h > 150:
            # move away
            print("move backward")
            self.bebop.fly_direct(0,-30,0,0,1)


def demo_user_code_after_vision_opened(bebopVision, args):
    bebop = args[0]

    
    bebop.smart_sleep(5)

    if (bebopVision.vision_running):
        # land
        bebop.safe_land(5)

        print("Finishing demo and stopping vision")
        bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    print("disconnecting")
    bebop.disconnect()

if __name__ == "__main__":
    # make my bebop object
    bebop = Bebop()

    # connect to the bebop
    success = bebop.connect(5)

    if (success):
        # start up the video
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run=demo_user_code_after_vision_opened,
                                     user_args=(bebop, ))
        userVision = UserVision(bebopVision,bebop)

        #take off
        bebop.safe_takeoff(10)
        bebop.fly_direct(0,0,0,50,1)
        bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None) #calls save picture continuously


        bebopVision.open_video()

    else:
        print("Error connecting to bebop.  Retry")