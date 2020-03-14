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

THROTTLE_SPEED = 16.5

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
    
    #We may fail finding the lines, in which case we leave lateral control to last command state
    try:
        #Find the center observation for lateral control
        grouped_lines = FindAndGroupLines(center_finder_input, probabilistic = False):
        lane_center = LineDetector.LaneCenterFinder(center_finder_input, grouped_lines)

        #Feed to lateral control and get PWM to send to servo
        image_dimensions = center_finder_input.shape
        desired_pwm = PIDController.LateralPIDControl(lane_center, image_dimensions, current_duty_cycle, THROTTLE_SPEED)

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
    #print(type(desired_pwm))
    #print("Desired PWM = {}".format(desired_pwm))

#Create an h264 encoded video for raspian os
def create_video_writer(filename,frame_size,fps):
    fourcc = cv.VideoWriter_fourcc(*'DIV4')
    filename = filename + '.avi'
    #fourcc = cv.VideoWriter_fourcc(*'MJPG')
    #filename = filename + '.avi'
    return cv.VideoWriter(filename, fourcc, fps, frame_size)
    

main_loop_count = 0
def main_loop(step_count = 100):
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
    global THROTTLE_SPEED
    throttle_pwm.SetDutyCycle(THROTTLE_SPEED)

    global lane_error_count
    lane_error_count = 0

    global main_loop_count
    main_loop_count += 1


    #window_name = "CameraBuffer"
    #cv.namedWindow(window_name)

    camera_buffer = RobotCamera.CameraBuffer()
    RobotCamera.camera.start_recording(camera_buffer, 'bgr', resize=(320,240))
    RobotCamera.camera.start_recording('TrackDC_Video_{}.h264'.format(main_loop_count), splitter_port=2)
    #my_image = np.ones( (240,320), dtype=np.uint8)
    keypressed = None
    #count = 100

    #Create video writer
    #video_writer = create_video_writer( 'track_video_{}'.format(main_loop_count), (240,320), 30)
    #create_video_writer(filename,frame_size,fps):

    #Timing and loop
    timec1 = time.clock()
    timet1 = time.time()
    for i in range(step_count):
        if keypressed == 'q':
            break
        #my_image = camera_buffer.read()
        #cv.imshow(window_name,my_image)
        #keypressed = cv.waitKey(50)
        #print(keypressed)
        #time.sleep(1)
        RobotCamera.camera.wait_recording()
        my_image = camera_buffer.read()

        command_robot(my_image,lateral_pwm)
        #video_writer.write(my_image)        
        
    timec2 = time.clock()
    timet2 = time.time()
    #print("time.time",time.time(),time.clock())
    print("time.clock", timec2-timec1, "time.time",timet2-timet1)

    #Cleanup Phase
    #video_writer.release()
    RobotCamera.camera.stop_recording()
    RobotCamera.camera.stop_recording(splitter_port=2)
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

    global THROTTLE_SPEED

    test_loop_i = 0
    step_count = 100
    
    while True:
                
        if servo_motor.pi.wait_for_edge(SIGNAL_PIN,servo_motor.pigpio.RISING_EDGE,10.0):
            time.sleep(1)
            test_loop_i += 1
            if (test_loop_i%2) == 1:
                THROTTLE_SPEED = 16.5
                step_count = 450
            else:
                THROTTLE_SPEED = 17.0
                step_count = 150
            print("Calling main THROTTLE_SPEED = {}...".format(THROTTLE_SPEED))

            main_loop(step_count)


if __name__ == "__main__":
    time.sleep(1)
    servo_motor.Init()
    #main_loop()
    #single_run()
    test_loop()
    servo_motor.DeInit()

