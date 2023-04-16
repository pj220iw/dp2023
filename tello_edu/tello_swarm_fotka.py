from djitellopy import Tello
import cv2
import socket
import threading
import time
#import libh264decoder
"""
tello = Tello('192.168.0.110', 8889)
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()
print('robím fotku')
cv2.imwrite("tello.jpg", frame_read.frame)

tello.disconnect()
"""
"""
i = 0
scale = 3

def TakePhoto(tello):
    tello.streamoff()
    tello2.streamoff()
    tello.streamon()
    tello2.streamon()
    img = tello.get_frame_read().frame
    img2 = tello2.get_frame_read().frame
    height, width, layers = img.shape
    height2, width2, layers2 = img2.shape
    new_h=int(height/scale)
    new_w=int(width/scale)
    new_h2=int(height2/scale)
    new_w2=int(width2/scale)
    resize = cv2.resize(img, (new_w, new_h))
    resize2 = cv2.resize(img2, (new_w2, new_h2))
    print('taking pciture')
    cv2.imwrite("tello1.jpg", resize)
    cv2.imwrite('tello2_%d', i, resize2)
"""
"""
while(True):
    ret, frame = telloVideo.read()
    if(ret):
        height, width, layers = frame.shape
        new_h = int(height/scale)
        new_w = int(width/scale)
        resize = cv2.resize(frame, (new_w, new_h))

        cv2.imshow('Tello', resize)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("tello1.jpg", resize)
            print("Take picture")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
"""

    

#tello2 = Tello('192.168.0.110', 8889)

#tello2.connect()
#tello.takeoff()
#tello2.takeoff()

#TakePhoto(tello)
#tello.release()
#cv2.destroyAllWindows()
#tello2.release()

#tello.land()
#tello2.land()