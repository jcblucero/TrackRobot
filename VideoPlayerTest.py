import numpy as np
import cv2 as cv

cap = cv.VideoCapture('lowres.h264')

while(cap.isOpened() ):
    ret, frame = cap.read()

    cv.imshow('frame',frame)
    #print(frame
    
    if cv.waitKey(50) & 0xFF == ord('q'):
        break
    
cap.release()
