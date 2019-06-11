#This file provides the Servo Motor class which controls
#servo motors through PWM on raspi GPIO

#Import GPIO library
import pigpio

import time
from subprocess import call

pi = None

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


#This class is to control servo motors
class ServoMotor:

    NUETRAL = 15 #15% duty cycle is nuetral on traxxas servos
    
    #pigpio expects duty cycle as a number out of 1 million
    #This class takes as input duty cycle as percentage from 0.0-100.0
    #multiply by 10k to convert
    DUTY_CYCLE_MULTIPLIER = 10000

    #Inputs: gpio_pin - one of PWM capable GPIO pins TODO: check valid pin
    #   frequency - frequency of pwm in hertz
    def __init__(self,gpio_pin,frequency):
        #Save input variables
        self.gpio_pin = gpio_pin
        self.frequency = frequency
        #init duty cycle to 15% (nuetral)
        self.duty_cycle = self.NUETRAL

        #Get reference to global scope pi var
        global pi
        if pi is None:
            PrintInitError()
            return
        
    #Update the duty cycle to the input duty cycle
    #Inputs: duty_cycle - float between 0.0-100.0
    def SetDutyCycle(self,duty_cycle):
        self.duty_cycle = duty_cycle
        pi.hardware_PWM(self.gpio_pin,self.frequency,self.duty_cycle*self.DUTY_CYCLE_MULTIPLIER)

        

        
