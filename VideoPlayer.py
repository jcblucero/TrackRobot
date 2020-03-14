import numpy as np
import cv2 as cv
import sys

class VideoPlayer(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.cap = cv.VideoCapture(self.file_name)

        #Make sure the file is valid
        if( self.cap.isOpened() == False):
            raise NameError("Could not find filename {}".format(self.file_name))
     
    def __del__(self):
        self.cap.release()

    #Returns next frame of video as an image (np.array) 
    def next_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    #Plays the whole video using cv.imshow
    #stops when user hits q
    def play_all(self):
        while(self.cap.isOpened() ):
            ret, frame = self.cap.read()

            cv.imshow('frame',frame)
            #print(frame

            #key = cv.waitKey(5)
            if cv.waitKey(50) & 0xFF == ord('q'):
                break       

    #Continuosly loop until the specified key is pressed, then return
    #if q is pressed, it will return (quit functionality). Cannot use q as input
    def wait_for_key(self,key_char):
        keystroke = cv.waitKey(10)
        while (keystroke & 0xFF) != ord(key_char):
            if(keystroke & 0xFF == ord('q')):
                return 'q'
            keystroke = cv.waitKey(50)
            #Do nothing

        return 'none'

    #Display a frame using cv.imshow and wait 50ms for frame to appear
    #returns true if q pressed
    def show_frame(self,frame):
        cv.imshow('frame',frame)
        if cv.waitKey(50) & 0xFF == ord('q'):
            return True
        else:
            return False     
"""
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

"""
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error, no input file specified")
        exit(0)
    else:
        file_name = sys.argv[1]

    video_player = VideoPlayer(file_name)
    #video_player.play_all()
    for i in range(10):
        ret, next_frame = video_player.next_frame()
        cv.imshow('frame',next_frame)
        ret_val = video_player.wait_for_key('n')
        if ret_val == 'q':
            break



