#Author: Jacob Lucero
#This file will loop infinitely asking for user input
#user inputs duty cycle to drive motor at (initialized to 15%duty cycle)
#when user enters to quit it stops

import servo_motor as SV

MOTOR_GPIO_PIN = 13 #GPIO19 is pin 35, can be used for PWM
TRAXXAS_PWM_FREQUENCY = 100 #100hz frequency for traxxas servos

#Create motor controller, and set to nuetral
SV.Init()
motor_pwm = SV.ServoMotor(MOTOR_GPIO_PIN,TRAXXAS_PWM_FREQUENCY)
motor_pwm.SetDutyCycle(motor_pwm.NUETRAL)

input_from_user = "none"
while input_from_user!="q":
    print("-----")
    print(input_from_user)
    try:
        new_duty_cycle = float(input_from_user)
        if new_duty_cycle>=0.0 and new_duty_cycle<=100.0:
            motor_pwm.SetDutyCycle(new_duty_cycle)
    except ValueError:
        print("Not a float")
    
    #print out current duty cycle and gather new input from user    
    print("motor set to {}".format(motor_pwm.duty_cycle) )
    input_from_user = raw_input("Enter a duty cycle or q to quit\n")

    

SV.DeInit()
