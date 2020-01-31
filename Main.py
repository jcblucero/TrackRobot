import cv2 as cv
import numpy as np
import math as math
import time


#Local Modules
#import RobotCamera
import LineDetector
import PIDController
import servo_motor
#TODO: Servo motors

MOTOR_GPIO_PIN = 19 #GPIO19 is pin 35, can be used for PWM
TRAXXAS_PWM_FREQUENCY = 100 #100hz frequency for traxxas servos

def main():

    #Init Servo and set to nuetral
    servo_motor.Init()
    lateral_pwm = servo_motor.ServoMotor(MOTOR_GPIO_PIN,
                                         TRAXXAS_PWM_FREQUENCY)
    lateral_pwm.SetDutyCycle(lateral_pwm.NUETRAL)
    current_duty_cycle = lateral_pwm.NUETRAL

    lane_error_count = 0

    #Global Input Files (for testing)
    input_filename = 'InputImages/low_res_pic_20.jpg'
    #input_filename = 'InputImages/trackpic12.jpg'
    output_filename = 'OutputImages/output.jpg'
    output_folder = 'OutputImages/'

    #Mimic input from video feed for now
    img = cv.imread(input_filename)
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    center_finder_input = cv.pyrDown(src=img_gray)

    #We may fail finding the lines, in which case we leave lateral control to last command state
    try:
        #Find the center observation for lateral control
        lane_center = LineDetector.LaneCenterFinder(center_finder_input)

        #Feed to lateral control and get PWM to send to servo
        image_dimensions = center_finder_input.shape
        desired_pwm = PIDController.LateralPIDControl(lane_center, image_dimensions, current_duty_cycle)

        #Set turning
        #TODO: clip this so it is within required range
        lateral_pwm.SetDutyCycle(desired_pwm)

    except Exception as error:
        #Increment error and report
        lane_error_count += 1
        print("Lane error count: {} ".format(lane_error_count) + repr(error) )

        desired_pwm = current_duty_cycle

        #Save a snapshot of when we failed
        if lane_error_count <= 5:
            cv.imwrite(output_folder + "lane_error_{}.jpg".format(lane_error_count),center_finder_input)

    #Output for testing
    print(type(desired_pwm))
    print("Desired PWM = {}".format(desired_pwm))

    



if __name__ == "__main__":

    main()
