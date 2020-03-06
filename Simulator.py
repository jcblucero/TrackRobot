import numpy as np
import math as math
import time

#Local Imports
import PIDController

MILES_TO_METERS_MULTIPLIER = 1609.344

#Models the steering servo
#Steering servo range is of 70deg from 55-125 assuming car travels along +y-axis
#and +x-axis is right, negative x-axis is left
#pwm < 15 = turn right
#pwm > 15 = turn left
#Output steering angle in radians
def model_steer_servo(pwm):

    #y=mx+b solved for where:
    #(10,55),(20,125)
    turn_angle_deg = 7*pwm - 15
    return turn_angle_deg * (np.pi/180.)

#Model throttle of traxxas on half speed setting (0-10mph assumed)
#Through testing, minimum to move is 16.4% pwm
#input - pwm (duty cycle % out of 100)
#Output velocity in meters/second
def model_halfspeed_throttle(pwm):

    #Assume a linear equation y=mx+b
    #where movement does not occur until 16.0% --/
    velocity_mph = (pwm - 16) * 2.5 
    velocity_mps = velocity_mph * MILES_TO_METERS_MULTIPLIER / 3600

    return velocity_mps

class RobotModel:

    def __init__(self,x_=0,y_=0,v_=0,theta_=0):
        self.x = x_
        self.y = y_
        self.velocity = v_
        self.steer_angle = theta_
        self.pwm = 0.

    #Update to next position based on commanded steering angle and speed
    #steer angle in radians
    #v - speed in meters/seconds
    #Input - time_step: time in seconds to move
    #Output (x,y) coordinates
    def move(self, time_step):
        self.x = self.x + (time_step * self.velocity) * np.cos(self.steer_angle)
        sefl.y = self.y + (time_step * self.velocity) * np.sin(self.steer_angle)


    #Execute one timestep of robot motion    
    def step(self):
    
        #TODO: need a way to scale x,y position of robot to image frame (240,320) pixels
        #TODO: measure width of camera and length of camera Field Of View
        #calculate error and get lateral control pwm
        lateral_pwm_command = PIDController.LateralPIDControl(
                                        measured_point=(self.x,self.y),
                                        image_dimensions=(240,320),
                                        current_duty_cycle = self.pwm
                                        )
        throttle_pwm_command = 16.4
        
        #Convert pwm's to steer_angle/velocity
        self.steer_angle = model_steer_servo(lateral_pwm_command)
        self.velocity = model_halfspeed_throttle(throttle_pwm_command)

        #Now update x/y position
        self.move()

    def plot():
    #TODO: matplotlib plot x/y position with x/y axis as 1 meter

#Simulate robot motion using PID control
def simulate(timesteps=100):

    robot_model = RobotModel()   
 
    for i in range(timesteps):
        
        robot_model.step()




