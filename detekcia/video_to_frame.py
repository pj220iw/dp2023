import cv2

def video_to_frames():
    vidcap = cv2.VideoCapture('tello/video/zahrada_vecer.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("tello/video_to_frames/12_04_2023/zahrada_ohen_vecer_%d.jpg" % count, image)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1