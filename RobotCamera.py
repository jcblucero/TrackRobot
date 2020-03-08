#
# This module is made to interact with raspberry pi camera V2 HW
#   Initializes camera to desired mode and settings
#   Provides mechanism to return camera image as numpy array
#   
# Current Config: Resolution 320x240p, Framerate=40fps, sensor mode 4 (full Field of View)
# Output is 1d np array of uint8 in grayscale
#
# TODO: provide different configuration settings (color output, image size, etc)

import cv2 as cv
import numpy as np
import math as math
import time
import picamera

camera = picamera.PiCamera(
    sensor_mode=4,
    #resolution='320x240',
    resolution='640x480',
    framerate=40)

#Using as reference: https://raspberrypi.stackexchange.com/questions/58871/pi-camera-v2-fast-full-sensor-capture-mode-with-downsampling
"""
class MyOutput(object):
    def write(self, buf):
        # write will be called once for each frame of output. buf is a bytes
        # object containing the frame data in YUV420 format; we can construct a
        # numpy array on top of the Y plane of this data quite easily:
        y_data = np.frombuffer(
            buf, dtype=np.uint8, count=128*96).reshape((96, 128))
        # do whatever you want with the frame data here... I'm just going to
        # print the maximum pixel brightness:
        print(y_data[:90, :120].max())

    def flush(self):
        # this will be called at the end of the recording; do whatever you want
        # here
        pass
"""
class CameraBuffer(object):

    def __init__(self):
        #self.data = np.empty( (240,320), dtype=np.uint8)
        self.data = np.zeros( (240,320), dtype=np.uint8)
    
    def write(self,buffer):
        #TODO: May have to lock this with mutex so that it isn't overwritten while copying
        #print("Write Called")
        #timet1 = time.time()
        temp = np.frombuffer( buffer, dtype=np.uint8, count=240*320*3).reshape( (240,320,3) )
        self.data = cv.cvtColor(temp,cv.COLOR_BGR2GRAY)
        #timet2 = time.time()
        #print(timet2-timet1)

    def read(self):
        return self.data

    def flush(self):
        pass


#May be useful to simplify using camera
class ImageFeed(object):

    def __init__(self):
        self.camera = picamera.PiCamera(
            sensor_mode=4,
            resolution='320x240',
            framerate=40)

        self.camera_buffer = CameraBuffer()

    def start(self):
        self.camera.start_recording(self.camera_buffer, 'rgb')
    
    def read(self):
        self.camera.wait_recording()
        return self.camera_buffer.read()


def main():
    window_name = "CameraBuffer"
    cv.namedWindow(window_name)

    camera_buffer = CameraBuffer()
    camera.start_recording(camera_buffer, 'bgr')
    #my_image = np.ones( (240,320), dtype=np.uint8)
    keypressed = None
    count = 3000
    
    for i in range(count):    
        my_image = camera_buffer.read()
        cv.imshow(window_name,my_image)
        keypressed = cv.waitKey(50)
        #print(keypressed)
        #time.sleep(1)
        camera.wait_recording()

    camera.stop_recording()
        
#Test harness for checking camera by itself
if __name__ == "__main__":
    main()



