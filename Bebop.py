import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import cv2
import cv_bridge
import dlib
import time

FLIGHT_TIME = 20
FACE_REC_INTERVAL = .1
FRAME_WIDTH = 428
FRAME_HEIGHT = 240

bridge = None
face_detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
last_image = None

drone_pub = None

def main():
    global last_image, drone_pub
    rospy.init_node("FaceTracker")
    camera_sub = rospy.Subscriber("/bebop/image_raw", Image, img_callback)
    drone_pub = rospy.Publisher("/bebop/cmd_vel", Twist, queue_size=1)
    takeoff_pub = rospy.Publisher("/bebop/takeoff", Empty, queue_size=1)
    landing_pub = rospy.Publisher("/bebop/land", Empty, queue_size=1)

    while last_image is None:
        time.sleep(0.5)

    takeoff_pub.publish(Empty())
    time.sleep(2.)

    last_call = 0
    start_time = time.time()
    while not rospy.is_shutdown() and time.time() - start_time < FLIGHT_TIME:
        if time.time - last_call > FACE_REC_INTERVAL:
            last_call = time.time()
            face_position = find_faces(last_image)
            print face_position

            adjust_drone_pos(face_position)

    landing_pub.publish(Empty())

    print("Shutdown.")

    cv2.destroyAllWindows()

    if __name__ == '__main__':
        bridge = cv_bridge.CvBridge()
        main()

def img_callback(img_msg):
    global bridge, last_image
    unscaled_cv_image = bridge.imgmsg_to_cv2(img_msg, "mono8")
    cv_image = cv2.resize(unscaled_cv_image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    last_image = cv_image

def find_faces(cv_image):
    global face_detector, win, last_call

    face_position = [None, None]

    faces = face_detector(cv_image, 1)

    print("Detections: {}".format(len(faces)))
    for i, d in enumerate(faces):
        print("face {}: Left: {}, Top: {}, Right: {}, Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        if i == 0: 
            face_position = (.5 * (d.right() + d.left()) / FRAME_WIDTH, .5 * (d.bottom() + d.top()) / FRAME_HEIGHT)

    rects = dlib.rectangles()
    rects.extend([d for d in faces])
    win.clear_overlay()
    win.set_image(cv_image)
    win.add_overlay(rects)

    # cv2.imshow("camera_raw", cv_image)
    # cv.2waitKey(3)

    return face_position

def adjust_drone_pos(face_pos):
    global drone_pub

    if face_pos[0] is None: return

    pos_update = Twist()

    if face_pos[0] < 0.3:
        # Turn CCW
        print "CCW"
        pass
    elif face_pos[0] > 0.7:
        # Turn CW
        print "CW"
        pass
    
    if face_pos[1] < 0.3: 
        # increase altitude
        print "ascend"
        pass

    elif face_pos[1] > 0.7:
        # reduce altitude
        print "descend"
        pass

    drone_pub.publish(pos_update)
    