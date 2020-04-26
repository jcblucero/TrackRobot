# TrackRobot
RC Car to follow lines on a track

#Introduction
The goal of this project is to create a robot that can follow the line around a 400 meter outdoor track and pace a runner. 

## Motivation
Often runners want to run 400 meters around a track at a specific pace, because different paces target different metabolic processes. 
However, it can be very difficult to hit your prescribed pace. Run too fast and you will burnout or overwork, run too slow and you will not get the designated physical adaptations of the workout.
This robot would help runners hit their target pace. By setting the robot at the desired pace/400m the runner could just follow the robot and know their pace is correct.

## Brief System Overview
The robot uses an RC car as the chassis. A camera is mounted to the front in order to view the track and it's lane lines. 
The camera is connected to a controller (raspberry pi), that does the image processing to find the lines.
The controller determines the center of the line and centers on this point. 
It centers itself using a PID controller, where error is calculated as the difference between the image center and the lane lines center.
The error is translated into a PWM to send to the steering servo.
The speed is controlled by sending a PWM to the Electronic Speed Control (ESC) of the RC car. The ESC then sets the brushed DC motor to correct speed.


Block Diagram of System

#Parts List
Traxxas Rustler XL-5(RC Car) with:
    Brushed DC Motor
    Electron Speed Control (ESC)
    Steering Servo
    4200 mAh, 8.4 V, niMH battery

Raspberry Pi 3b+
Raspberry Pi Camera Module v2
16000 mAh, 2.4v Output, USB charger (to power raspberry pi)


#Controlling Traxxas RC Car
In order to get the RC car moving, and traveling in the right direction, I had to understand how the steering and throttle motors were controlled.
Under normal operation, the steering servo and ESC are plugged into the radio transceiver. The transceiver gets the radio control signal and transfers it over the wire to the ESC and steering motor.

I first looked at the ESC. It has 3 wires connected to the radio transceiver: power, ground, and signal. I connected the signal wire up to an oscilliscope to view waveform.
I found a 100 Hz (10 milisecond) PWM signal that had a duty cycle with a range of 10%-20%, and voltage levels of 0-3v. 20% duty cycle indicated full throttle, 10% indicated full brake, and 15% indicated nuetral. 
The ESC has multiple modes according to the user manual. If in a mode with reverse, then 10% will indicate reverse if coming from a nuetral (15%) position.
The steering motor uses the same signal (0-3v, 10-20% duty cycle, 100Hz) where 10% commands full left steering and 20% full right.

Below image shows the signal in a nuetral position.
