#Import GPIO library
import pigpio
import picamera
import time
from subprocess import call

pi = None
SIGNAL_PIN = 19
picamera = picamera.PiCamera()

def PrintInitError():
    print("Error: servo_motor.py is not initialized. Call servo_motor.Init() to init, and servo_motor.DeInit() when finished")
    
def Init():
    global pi
    #Make sure pigpio daemon is running
    call(["sudo","pigpiod"])
    #give daemon time to start
    time.sleep(0.5)

    #Init pigpio to this raspberry pi
    pi = pigpio.pi()

def DeInit():
    global pi
    if pi is None:
        PrintInitError()
        return
    pi.stop()
    call(["sudo","pkill","pigpiod"])

picture_count = 0
def TakePicture_PigpioCallback(gpio,level,tick):
    global picture_count
    picture_count += 1
    print("picture taken: {}".format(picture_count))
    #time.sleep(3)
    
def CapturePicture():
    global picture_count
    picture_count += 1
    print("picture taken: {}".format(picture_count))
    picamera.capture('trackpic{}.jpg'.format(picture_count))
    picamera.start_recording('trackvid{}.h264'.format(picture_count))
    time.sleep(4)
    picamera.stop_recording()
                             

#Take pictures when GPIO Pin is set pulled to ground#
Init()
pi.set_mode(SIGNAL_PIN, pigpio.INPUT)
pi.set_pull_up_down(SIGNAL_PIN,pigpio.PUD_DOWN)
#set callback to Rising edge (pulled low, then connected to 3.3v to activate)
#callback_object = pi.callback(SIGNAL_PIN,pigpio.RISING_EDGE,TakePicture_PigpioCallback)
    

while True:

    if pi.wait_for_edge(SIGNAL_PIN,pigpio.RISING_EDGE,10.0):
        CapturePicture()
        time.sleep(1)

    #Pulled down at start
    #if pi.read(SIGNAL_PIN) == 0:
    #    signal_low = True

    #while signal_low:
        #If it is pulled up, take picture
    #    if (pi.read(SIGNAL_PIN) == 1):
    #        CapturePicture()
    #        signal_low = False


####Main Loop#####
#input_from_user = "none"
#while input_from_user!="q":
#
#    input_from_user = raw_input("Press q to quit\n")

DeInit()




