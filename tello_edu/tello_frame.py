from djitellopy import Tello
import cv2
import time

tello = Tello('192.168.0.107', 8889)
tello.connect()
tello.takeoff()
tello.streamon()
count = 0
vidcap = cv2.VideoCapture(tello.get_udp_video_address())

success,image = vidcap.read()
success = True

while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
    success,image = vidcap.read()
    print ('Read a new frame: ', success2)
    cv2.imwrite("fotky/frame_2%d.jpg" % count, image2) # save frame as JPEG file
    count = count + 1