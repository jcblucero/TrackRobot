import numpy as np
import cv2 as cv

class VideoPlayer(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.cap = cv.VideoCapture(self.file_name)

    def __del__(self):
        self.cap.release()

    #Returns next frame of video as an image (np.array) 
    def next_frame(self):
        ret, frame = cap.read()
        return frame

    #Plays the whole video using cv.imshow
    #stops when user hits q
    def play_all(self):
        while(cap.isOpened() ):
            ret, frame = cap.read()

            cv.imshow('frame',frame)
            #print(frame

            #key = cv.waitKey(5)
            if cv.waitKey(50) & 0xFF == ord('q'):
                break       

#cap = cv.VideoCapture('TrackDC_Video_4.h264')
cap = cv.VideoCapture('TrackDC_Intercepting_Lines.h264')

while(cap.isOpened() ):
    ret, frame = cap.read()

    cv.imshow('frame',frame)
    #print(frame

    #key = cv.waitKey(5)
    if cv.waitKey(50) & 0xFF == ord('q'):
        break
    
cap.release()
