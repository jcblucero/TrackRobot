import cv2 as cv
import numpy as np
import math as math
import time


#Local Modules
import RobotCamera
import LineDetector
import PIDController
import servo_motor
#TODO: Servo motors

THROTTLE_GPIO_PIN = 18
STEERING_GPIO_PIN = 19 #GPIO19 is pin 35, can be used for PWM
TRAXXAS_PWM_FREQUENCY = 100 #100hz frequency for traxxas servos
output_folder = 'OutputImages/'

#Make sure we don't put egregious output to PWM
#Only valid values are 10.0-20.0
def clip_pwm(pwm):
    if pwm > 20.0:
        return 20.0
    elif pwm < 10.0:
        return 10.0
    else:
        return pwm

#Take input image and servo_motor object
#Process image to find line center
#Command lateral control based on center and current duty cycle
def command_robot(center_finder_input,lateral_pwm):

    current_duty_cycle = lateral_pwm.GetDutyCycle()
    global lane_error_count
    print(lane_error_count)
    
    #We may fail finding the lines, in which case we leave lateral control to last command state
    try:
        #Find the center observation for lateral control
        lane_center = LineDetector.LaneCenterFinder(center_finder_input)

        #Feed to lateral control and get PWM to send to servo
        image_dimensions = center_finder_input.shape
        desired_pwm = PIDController.LateralPIDControl(lane_center, image_dimensions, current_duty_cycle)

        #Set turning
        #TODO: clip this so it is within required range
        desired_pwm = clip_pwm(desired_pwm)
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

def main_loop():
    #Init Servo and set to nuetral
    #servo_motor.Init()
    lateral_pwm = servo_motor.ServoMotor(STEERING_GPIO_PIN,
                                         TRAXXAS_PWM_FREQUENCY)
    lateral_pwm.SetDutyCycle(lateral_pwm.NUETRAL)
    current_duty_cycle = lateral_pwm.GetDutyCycle()

    #Throttle pwm
    throttle_pwm = servo_motor.ServoMotor(THROTTLE_GPIO_PIN,
                                         TRAXXAS_PWM_FREQUENCY)
    #16.4% PWM is just enough to move when ESC in half speed mode
    throttle_pwm.SetDutyCycle(16.5)

    global lane_error_count
    lane_error_count = 0


    #window_name = "CameraBuffer"
    #cv.namedWindow(window_name)

    camera_buffer = RobotCamera.CameraBuffer()
    RobotCamera.camera.start_recording(camera_buffer, 'rgb')
    #my_image = np.ones( (240,320), dtype=np.uint8)
    keypressed = None
    count = 50

    #Timing and loop
    timec1 = time.clock()
    timet1 = time.time()
    for i in range(count):
        if keypressed == 'q':
            break
        my_image = camera_buffer.read()
        #cv.imshow(window_name,my_image)
        #keypressed = cv.waitKey(50)
        #print(keypressed)
        #time.sleep(1)
        RobotCamera.camera.wait_recording()

        command_robot(my_image,lateral_pwm)
        
    timec2 = time.clock()
    timet2 = time.time()
    #print("time.time",time.time(),time.clock())
    print("time.clock", timec2-timec1, "time.time",timet2-timet1)

    #Cleanup Phase
    RobotCamera.camera.stop_recording()
    throttle_pwm.SetDutyCycle(throttle_pwm.NUETRAL)
    lateral_pwm.SetDutyCycle(lateral_pwm.NUETRAL)
    #servo_motor.DeInit()
    
def single_run():

    #Init Servo and set to nuetral
    #servo_motor.Init()
    lateral_pwm = servo_motor.ServoMotor(STEERING_GPIO_PIN,
                                         TRAXXAS_PWM_FREQUENCY)
    lateral_pwm.SetDutyCycle(lateral_pwm.NUETRAL)
    current_duty_cycle = lateral_pwm.GetDutyCycle()

    global lane_error_count
    lane_error_count = 0

    #Global Input Files (for testing)
    #input_filename = 'InputImages/low_res_pic_20.jpg'
    input_filename = 'low_res_pic_1.jpg'
    #input_filename = 'InputImages/trackpic12.jpg'
    output_filename = 'OutputImages/output.jpg'
    output_folder = 'OutputImages/'

    #Mimic input from video feed for now
    img = cv.imread(input_filename)
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    center_finder_input = cv.pyrDown(src=img_gray)

    command_robot(center_finder_input,lateral_pwm)

    #Cleanup Phase
    lateral_pwm.SetDutyCycle(lateral_pwm.NUETRAL)
    #servo_motor.DeInit()


#Loop for testing on track
#Calls main_loop when GPIO 21 pulled up. Loop indefinitely
def test_loop():
    SIGNAL_PIN = 21
    servo_motor.pi.set_mode(SIGNAL_PIN, servo_motor.pigpio.INPUT)
    servo_motor.pi.set_pull_up_down(SIGNAL_PIN,servo_motor.pigpio.PUD_DOWN)
    
    while True:
                
        if servo_motor.pi.wait_for_edge(SIGNAL_PIN,servo_motor.pigpio.RISING_EDGE,10.0):
            print("Calling main...")
            main_loop()


if __name__ == "__main__":
    time.sleep(1)
    servo_motor.Init()
    #main_loop()
    #single_run()
    test_loop()
    servo_motor.DeInit()

