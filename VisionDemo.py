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
    def __init__(self, vision):
        self.index = 0
        self.vision = vision
        self.filename = "test_image.png"

    def save_pictures(self, args):
        #print("saving picture")
        img = self.vision.get_latest_valid_picture()

        if img is not None:   # saving latest img
            self.filename = "test_image.png" # % self.index
            cv2.imwrite(self.filename, img)
            self.index += 1
            self.detect_recognize_face()

    def detect_recognize_face(self):
        pil_image = Image.open("test_image.png").convert("L") # convert("L") changes img to grayscale
        #pil_image.show()
        final_image = pil_image.resize((550,550), Image.ANTIALIAS)
        image_array = np.array(final_image, "uint8")
        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=1) # higher scale facter might increase accuracy
        for (x, y, w, h) in faces:
            print(x,y,w,h)            
            print("Detected")
            roi_color = image_array[y:y+h, x:x+w]

            #recognizer
            id_, conf = recognizer.predict(roi_color)
            if conf <= 70:  # and conf <= 85:  # 0 is perfect match  200 is max i guess?
                print(id_)
                print(labels[id_])
                self.adjust_drone_pos(x,y,w,h)

                # if id_ == 1:
                #     return 1
                # elif id_ == 2:
                #     return 2
                # elif id_ == 3:
                #     return 3
                # else:
                #     return 999
                #drone movement goes in here

    def check_5sec(self):
        t_end = time.time() + 5
        id_arr = []
        for i in range(0,len(labels)):
            id_arr.append(0)
        while time.time() < t_end:
            id = self.detect_recognize_face()
            id_arr[id] += 1

        idmax = max(id_arr)

        return idmax

    def perform_action(self,id):
        if id == 1:
            #bebop.flip(back)
            bebop.smart_sleep(5)
            print("backflip")

        elif id == 2:
            # bebop.flip(front)
            bebop.smart_sleep(5)
            print("frontflip")
        elif id == 3:
            # bebop.turn_degrees(180)
            bebop.smart_sleep(5)
            print('turn around')
        else:
            #ATTACK
            print("attack")
    
        self.safe_land(10)

    def adjust_drone_pos(self, x, y, w, h):

        if x == None:
            return
        
        if x < 100:
            print("turn CCW")
            # bebop.turn_degrees(-5)
            # bebop.smart_sleep(2)

        if x > 350:
            print("turn CW")
            # bebop.turn_degrees(5)
            # bebop.smart_sleep(2)

        if y < 100:
            print("Ascend")

        if y > 350:
            print("descend")

        if h < 75:
            print("Move Forward")
        
        if h > 150:
            print("Move Backward")




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

#def something(bebop):

if __name__ == "__main__":
    # make my bebop object
    bebop = Bebop()

    # connect to the bebop
    success = bebop.connect(5)

    if (success):
        # start up the video
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run=demo_user_code_after_vision_opened,
                                     user_args=(bebop, ))
        userVision = UserVision(bebopVision)
        #take off
        #bebop.safe_takeoff(10)
        bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None) #calls save picture continuously
        #bebopVision.set_user_callback_function(something(bebop), user_callback_args=None) #calls save picture continuously

        frame = bebopVision.get_latest_valid_picture()

        bebopVision.open_video()

    else:
        print("Error connecting to bebop.  Retry")