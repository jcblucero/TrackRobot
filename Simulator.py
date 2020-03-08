import numpy as np
import math as math
import matplotlib.pyplot as plt
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

#Rough estimate of how x,y position translates to pixel position in camera
#Made using empirical measurments: mid-y = .33meters wide, end-y = .7112 meters wid
#                               y-length = 1 meter
# Transform assuming we are measuring center at mid-y point
#Input - (x,y) location to transorm
#Output - (x,y) with x translated
#TODO: look up camera transform/projection/rotation matrix and figure out real way
def camera_transform(point):
    x = point[0]
    y = point[1]
    x = (x / 0.33) * 320
    return (x,y)

class RobotModel:

    def __init__(self,x_=0,y_=0,v_=0,theta_=0):
        self.x = x_
        self.y = y_
        self.velocity = v_
        self.steer_angle = theta_
        self.pwm = 0.

        self.x_series = []
        self.y_series = []

    #Update to next position based on commanded steering angle and speed
    #steer angle in radians
    #v - speed in meters/seconds
    #Input - time_step: time in seconds to move
    #Output (x,y) coordinates
    def move(self, time_step):
        self.x = self.x + (time_step * self.velocity) * np.cos(self.steer_angle)
        #y is always moving forward (forward dir of car), so we want the magnitude, not sign
        self.y = self.y + np.abs((time_step * self.velocity) * np.sin(self.steer_angle))


    #Execute one timestep of robot motion
    #One timestep = 1) Get lateral error (based on x,y pos) and pwm command. 
    # 2) Update steer_angle and velocity based models of servos
    # 3) move robot
    def step(self):
    
        #TODO: need a way to scale x,y position of robot to image frame (240,320) pixels
        #TODO: measure width of camera and length of camera Field Of View
        #calculate error and get lateral control pwm
        robot_position = (self.x,self.y)
        image_point = camera_transform(robot_position)
        print("Current Position ",(robot_position), "Image position ", image_point)
        lateral_pwm_command = PIDController.LateralPIDControl(
                                        #measured_point=(self.x,self.y),
                                        measured_point = image_point,
                                        image_dimensions=(240,320),
                                        current_duty_cycle = self.pwm
                                        )
        throttle_pwm_command = 17.2
        
        #Convert pwm's to steer_angle/velocity
        self.steer_angle = model_steer_servo(lateral_pwm_command)
        self.velocity = model_halfspeed_throttle(throttle_pwm_command)

        #Now update x/y position
        self.move(0.2)

        #Capture in time series for plotting
        self.x_series.append(image_point[0])
        self.y_series.append(self.y)

    def plot(self):
    #TODO: matplotlib plot x/y position with x/y axis as 1 meter
        print(self.x_series)
        plt.figure()
        plt.title("Robot Position")
        plt.plot(self.x_series,self.y_series,'o-')
        plt.show()

#Simulate robot motion using PID control
def simulate(timesteps=20):

    robot_model = RobotModel(x_=1)   
 
    for i in range(timesteps):
        
        robot_model.step()

    robot_model.plot()

if __name__ == "__main__":
    simulate()


