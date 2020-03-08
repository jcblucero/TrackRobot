import cv2 as cv
import numpy as np
import math as math
import time


#Calculate the raw error which will eventually be used by PID controller for steering
#left/right steering only, so only 1d input
def CalculateRawError(measured, desired):
    return desired - measured

#Calculate the error which will be used to feed PID controller for steering
#We use a scaled error between [-100.0,100.0] (float) so that PID is impartial to viewing dimensions
#Inputs: center_point - (x,y) pair  - center of lines found through line detection
#       image_dimensions - tuple - dimensions of image fed by camera to line detection
#Outputs: Float between [-100.0,100.0] representing error.
#by convention, positive means measured center is Robot left of image center(image right),so we want to turn left
# < 15 turns right. > 15 turns left when output to servo pwm.
def CalculateScaledTrajectoryError( center_point, image_dimensions):
    
    #print("Image Dimensions {}".format(image_dimensions))

    #We care about lateral error,since we are only steering left/right.
    #This means x direction (columns) of image
    measured_value = center_point[0] #x index is 0
    desired_value = int(image_dimensions[1] / 2) #TODO: check this index - Should be lateral (x/columns) dimension, works on RGB/Grayscale
    max_error = desired_value #TODO: check this index

    raw_error = float(measured_value - desired_value)

    scaled_error = (raw_error / max_error) * 100.0
    #now clip scaled_error so it is in range [-100,100] incase we get way off
    scaled_error = max(min(scaled_error,100.0),-100.0)

    #print("Max Error = {}, Raw Error = {}, Scaled Error = {}".format(max_error, raw_error,scaled_error) )

    return scaled_error

# PWM duty cycle must be between 10-20% for servo motors (10-20ms) with 15% indicating nuetral (no turn/no throttle)
# We need a linear function to normalize to range 10.0-20.0. Let x=percent error, and y = requested duty cycle
# FULL LEFT ERROR:  (-100.0,10.0)
# NUETRAL:          (0.0, 15.0)
# FULL RIGHT ERROR: (100.0, 20.0)
# Solving Y=mX+b we get: Y = 0.05 * X + 15
# Input:
#   percent_error - float in range [-100.0,100.0]
# Output:
#   duty_cycle - float in range [10.0-20.0] - duty cycle to feed to servo motor for throttle/steering
def NormalizeErrorToServoRange(percent_error):
    duty_cycle = 0.05 * percent_error + 15
    return duty_cycle

# Determine what PWM duty cycle (as percent) to command to steering servo for lateral control
# Inputs:
#   measured_point - (x,y) pair  - center of lines found through line detection
#   image_dimensions - tuple - dimensions of image fed by camera to line detection
#   current_duty_cycle - float - current PWM duty cycle on controller
#   last_error - float - error from last call to LateralPIDControl. Used for Derivative calc
def LateralPIDControl( measured_point, image_dimensions, current_duty_cycle):

    Kp = 1.0

    scaled_error = CalculateScaledTrajectoryError(measured_point,image_dimensions)    

    #Proportional Control
    #TODO: Future add Integral and Derivative for PID control
    scaled_error = scaled_error * Kp

    #Normalize to the servo range for output
    return NormalizeErrorToServoRange(scaled_error)
    

