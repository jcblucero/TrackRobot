import cv2 as cv
import numpy as np
import math as math
import time


#Local Modules
#import RobotCamera
import LineDetector
import PIDController
#TODO: Servo motors

def main():

    #Global Input Files (for testing)
    input_filename = 'InputImages/low_res_pic_20.jpg'
    #input_filename = 'InputImages/trackpic3.jpg'
    output_filename = 'OutputImages/output.jpg'
    output_folder = 'OutputImages/'

    #Mimic input from video feed for now
    img = cv.imread(input_filename)
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    center_finder_input = cv.pyrDown(src=img_gray)

    #Find the center observation for lateral control
    lane_center = LineDetector.LaneCenterFinder(center_finder_input)

    #Feed to lateral control and get PWM to send to servo
    image_dimensions = center_finder_input.shape
    current_duty_cycle = 15.0
    desired_pwm = PIDController.LateralPIDControl(lane_center, image_dimensions, current_duty_cycle)

    #Output for testing
    print("Desired PWM = {}".format(desired_pwm))



if __name__ == "__main__":

    main()
