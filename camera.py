from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI
import threading
import cv2
import time
import pickle
import numpy as np
from PIL import Image
from PIL import ImageFile


face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

cap = cv2.VideoCapture(0)
d = 0
class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision
        self.filename = "test_image.png"

    def save_pictures(self, args):
        #print("saving picture")
        img = self.vision.get_latest_valid_picture()

        if img is not None:   # saving latest img
            filename = "images/file_%d.jpg" % self.index
            cv2.imwrite(filename, img)
            self.index += 1

def demo_user_code_after_vision_opened(bebopVision, args):
    bebop = args[0]

    print("Vision successfully started!")
    #removed the user call to this function (it now happens in open_video())
    #bebopVision.start_video_buffering()

    # takeoff
    #bebop.safe_takeoff(5)

    # skipping actually flying for safety purposes indoors - if you want
    # different pictures, move the bebop around by hand
    print("Fly me around by hand!")
    bebop.smart_sleep(5)

    if (bebopVision.vision_running):
        print("Moving the camera using velocity")
        bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-2, duration=4)
        bebop.smart_sleep(5)

        # land
        #bebop.safe_land(5)

        print("Finishing demo and stopping vision")
        bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    print("disconnecting")
    bebop.disconnect()
while True:
    bebop = Bebop()

    # connect to the bebop
    success = bebop.connect(5)

    if (success):
        # start up the video
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run=demo_user_code_after_vision_opened,
                                     user_args=(bebop,))
        userVision = UserVision(bebopVision)
        bebopVision.set_user_callback_function(userVision.save_pictures,
                                               user_callback_args=None)  # calls save picture continuously

        frame = bebopVision.get_latest_valid_picture()

        bebopVision.open_video()

    else:
        print("Error connecting to bebop.  Retry")



