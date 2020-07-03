import cv2 as cv
import numpy as np
import math as math
import time
import VideoPlayer

#Global Output folder (for testing)
output_folder = 'OutputImages/'

#Colors for drawing on image
green = [0,255,0]
purple = [255,0,255]
blue = [255,0,0]

### Classes

class YInterceptLine(object):
    #Line represented by: y = m*x+b

    def __init__(self, _m, _b):
        self.m = _m
        self.b = _b
    
    #Predict the y value
    def predict_y(self, x):
        #y = m*x + b
        return (self.m * x) + self.b

    #Predict the x value
    def predict_x(self, y):
        #x = (y-b) / m
        return (y - self.b) / self.m

    #Draw this line on given image
    def draw(self, img):
        color = [0,255,0]
        left_x = 0
        right_x = img.shape[1] #columns
        pt1 = (left_x, int(self.predict_y(left_x)) )
        pt2 = (right_x, int(self.predict_y(right_x)) )
        cv.line( img, pt1, pt2, color, 2)

class PolarLine(object):
    #Line represented by: r = x *cos(theta) + y*sin(theta)

    def __init__(self, _rho, _theta):
        self.rho = _rho
        self.theta = _theta
    
    #Predict the y value
    def predict_y(self, x):
        #Y = (r-Xcos(theta)) / sin(theta)
        return (self.rho - (x*np.cos(self.theta))) / np.sin(self.theta)

    #Given y, predict the x value
    def predict_x(self, y):
        #X = (r-Ysin(theta)) / cos(theta)
        return (self.rho - (y*np.sin(self.theta))) / np.cos(self.theta)

    #Draw this line on given image
    def draw(self, img):
        color = [0,255,0]
        left_x = 0
        right_x = img.shape[1] #columns
        pt1 = (left_x, int(self.predict_y(left_x)) )
        pt2 = (right_x, int(self.predict_y(right_x)) )
        #print("pt1,pt1",pt1,pt2)
        cv.line( img, pt1, pt2, color, 2)


#This is a pipeline that applies dilate, gaussian blur, then a canny filter to output edges.
#Tuned for 320x240 image
#Input - 320x240 gray image (8 bit input)
#output - 320x240 edge image (1 for edge, 0 for not)
def FilterPipeline(img_gray):

    #dilation params
    #Dilation takes max within kernel, and applies it to every other pixel within kernel
    dilate_kernal_size = 5
    dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilate_kernal_size,dilate_kernal_size) )

    #Gauss kernal params
    gauss_kernel_size = 21
    gauss_kernel = (gauss_kernel_size,gauss_kernel_size)
    gauss_sigma = 3 #for both x and y

    #Canny filter params
    #Recommended upper is 2x-3x lower threshold (rec by openCV)
    lower_canny_threshold = 25
    upper_canny_threshold = 75
    canny_kernel_size = 3

    #Step 1 of Filter - Dilate
    filtered_img = cv.dilate(src=img_gray, kernel=dilate_kernel)

    #Step 2 of Filter - Gaussian Blur
    filtered_img = cv.GaussianBlur(
       src=filtered_img,ksize=gauss_kernel, sigmaX=gauss_sigma, sigmaY=gauss_sigma,\
       borderType=cv.BORDER_REPLICATE)
    #gauss_img = cv.GaussianBlur(img_gray,(5,5),0,0) #sigmaX/Y as 0's lets func determine
    #filtered_img = cv.medianBlur(img_gray,kernel_size)
    #cv.imwrite(output_folder + "filtered_image.jpg",filtered_img)

    #Step 3 of Filter - Canny Edge Detector
    #edges = cv.Canny(filtered_img,lower_canny_threshold,upper_canny_threshold)
    edges = cv.Canny(
        image=filtered_img,threshold1=lower_canny_threshold,threshold2=upper_canny_threshold, \
        apertureSize=canny_kernel_size)
    #cv.imwrite(output_folder + "edges.jpg",edges)

    return edges

#Outputs Houghlines tuned for 320x240 image
# Uses cv.HoughLinesP if probabilistic=True, which outputs lines as two (x,y) points
# default uses cv.HoughLines, which outputs lines in rho,theta format
def FindLines(edge_img, probabilistic=False):
    rho_resolution = 1 #distance resolution of accumulator in pixels
    theta_resolution = np.pi / 180 #angle resolution of accumulator in rads
    threshold = 50 #only return lines with greater number of votes

    #for HoughLInesP
    min_line_length = 50#100
    max_line_gap = 50#100

    if probabilistic:
        lines = cv.HoughLinesP(edge_img, rho=rho_resolution, theta=theta_resolution, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    else:
        lines = cv.HoughLines(edge_img,rho_resolution,theta_resolution,threshold)#,min_theta=0,max_theta=2*np.pi)    

    return lines

# Take HoughLines and eliminate unlikely candidates. Then group using kmeans and output as np array
def FilterForTrackLaneLines(lines, probabilistic=False):

    #Raise an exception if there are less than the two edges of a single line. Clustering assumes atleast 2
    if lines.shape[0] < 2:
        raise Exception('Expected atleast 2 lines before filter. Found {}'.format(lines.shape[0]))

    #Step 1: Take output of HoughLines(P) and convert to useful np matrix format
    if probabilistic:
        clustering_lines = np.zeros( (lines.shape[0],2) )
        count = 0
        for line in lines:
            #print("Line #{}".format(count) )
            coords = line[0]
            pt1 = (coords[0],coords[1])
            pt2 = (coords[2],coords[3])

            #Creating numpy matrix from tuple to feed to kmeans
            slope_intercept = ConvertToSlopeIntercept(coords)
            #print("Slope = {} , B = {} ".format(slope_intercept[0],slope_intercept[1]))
            clustering_lines[count] = np.array(slope_intercept)
            count += 1

    else:
        matrix_shape = (lines.shape[0],lines.shape[2])
        clustering_lines = lines.reshape(matrix_shape)

    #Step 2 - Eliminate lines that are unlikely to be track lane lines
    if probabilistic:
        #Filter lines that are too horizontal (+-30deg of 0 which is horizontal)
        #tan(theta) = opp/adj -> rise/run -> slope. tan(theta) == slope
        lower_angle_threshold = np.tan(30 * np.pi/180)
        upper_angle_threshold = np.tan(-30 * np.pi/180)

        selection_array = (clustering_lines[:,0] >= lower_angle_threshold) | (clustering_lines[:,0] <= upper_angle_threshold)
        clustering_lines = clustering_lines[selection_array] 
    else:
        #Filter lines that are too horizontal (60-120deg, +-30deg of 90 which is horizontal)
        lower_angle_threshold = 60 * np.pi/180
        upper_angle_threshold = 120 * np.pi/180

        selection_array = (clustering_lines[:,1] <= lower_angle_threshold) | (clustering_lines[:,1] >= upper_angle_threshold)
        clustering_lines = clustering_lines[selection_array] 

    #print ("filtered clustering lines")
    #print (clustering_lines)

    #Raise an exception if there are less than the two lines after filtering. Centering assumes atleast 2
    if clustering_lines.shape[0] < 2:
        raise Exception('Expected atleast 2 lane lines after filter. Found {}'.format(clustering_lines.shape[0]))

    #Step 3 - Cluster remaining lines so that we have 2 groups.
    if probabilistic:
        # probabilistic lines are in slope-intercept (m,b) format so cluster on that
        criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 0.001)
        clustering_lines = np.float32(clustering_lines)
        ret,label,best_lines=cv.kmeans(clustering_lines,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
                                     
    else:                                     
    # standard hough lines in polar(rho,theta). This has discontinuity on when going from 2pi->0...
    # ...so we convert to (x,y) representation and cluster on those points
        #Convert to x/y form using sin/cos to avoid discontinuities in (theta 0-2pi, and rho negative distance)
        xy_clustering = np.copy(clustering_lines)
        xy_clustering[:,0] = np.sin(clustering_lines[:,1]) * clustering_lines[:,0] #y's = sin(theta) * rho
        xy_clustering[:,1] = np.cos(clustering_lines[:,1]) * clustering_lines[:,0] #x's = cos(theta) * rho
        
        #kmeans clustering
        criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 0.001)
        xy_clustering = np.float32(xy_clustering)
        ret,label,centers_xy=cv.kmeans(xy_clustering,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

        #convert back to theta/rho
        best_lines=np.copy(centers_xy)
        best_lines[:,0] = np.sqrt(centers_xy[:,0]*centers_xy[:,0] + centers_xy[:,1]*centers_xy[:,1]) #rho
        #best_lines[:,1] = np.arctan(centers_xy[:,0]/centers_xy[:,1])
        best_lines[:,1] = np.arctan2(centers_xy[:,0],centers_xy[:,1])

        #print("KMeans Centers -- \n{}".format(best_lines))

    return best_lines

# make sure that any lines in given input, reach from the top of the image to bottom (within img dimensions)
# Input - prediction_lines - List of Yintercept/PolarLines
#   image_shape - shape of image to check dimensions of (rows,columns)
# Output - returns list of lines that fit within dimensions while spanning top to bot
def FilterForTopToBotLines(prediction_lines,image_shape):
    output = []
    left = 0
    right = image_shape[1]
    top = 1 #Don't start at 0 to give some leeway
    bot= image_shape[0] - 1 #again, bring in by 1 pixel to give more leeway
    #print(right)
    for line in prediction_lines:
        top_pred = line.predict_x(top)
        bot_pred = line.predict_x(bot)
        if (top_pred <= right) and (bot_pred <= right) and (top_pred >= left) and (bot_pred >= left):
            output.append(line)

    return output

#Predict the x position corresponding to y for each line, then take the middle of those two
def PredictLinesCenterX(y,prediction_lines):
    x_sum = 0
    for line in prediction_lines:
        x_sum += line.predict_x(y)

    center_point = ( int(x_sum / len(prediction_lines) ), y)
    
    return center_point
    

#Finds the center of two (x,y) points and returns (x,y) as integers
def FindCenter(pt1,pt2):
    center_x = int((pt1[0] + pt2[0]) / 2)
    center_y = int((pt1[1] + pt2[1]) / 2)
    return (center_x,center_y)

#Convert lines in represented by two points in form (x0,y0,x1,y1)
#to slope intercept form: y=mx+b represented by (m,b)
def ConvertToSlopeIntercept(line_coords):
    x0,y0 = (line_coords[0],line_coords[1])
    x1,y1 = (line_coords[2],line_coords[3])
    m = float((y1-y0)) / (x1-x0)
    #b = y-mx
    b = y0 - m*x0
    slope_intercept = (m,b)
    return slope_intercept


# Draw lines on image. Probabilistic tells us whether HoughLines or HoughLinesP was used to generate lines
# image - image to draw lines in
# lines - array of lines. output of Houghlines(P). Either np.array of (rho,theta) or np.array of (pt1.x,pt1.y,pt2.x,pt2.y)
# probabilistic - if true, lines are in point form
def DrawLines(img,lines,probabilistic):
    red = [0,0,255]
    width = 3

    if lines is None:
        print( "No lines found")
        return 

    if probabilistic:
        print("Lines Shape:")
        print(lines.shape)
        max_lines_to_print = 2
        count = 0
        for line in lines:
            print("Line #{}".format(count) )
            count += 1
            coords = line[0]
            pt1 = (coords[0],coords[1])
            pt2 = (coords[2],coords[3])
            cv.line(img, pt1, pt2, red, width)
            print("point 1 = {},{} ; point 2 = {},{}".format(pt1[0],pt1[1],pt2[0],pt2[1]))

            slope_intercept = ConvertToSlopeIntercept(coords)
            print("Slope = {} , B = {} ".format(slope_intercept[0],slope_intercept[1]))
    else:
        count = 0
        for line in lines:
            print("Line #{}".format(count) )
            count += 1
            ##Adapted from OpenCv HoughLines Tutorial
            rho = line[0][0]
            theta = line[0][1]
            print("Theta {}, Rho {}".format(theta,rho) )
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a*rho
            y0 = b*rho
            pt1 = ( int(x0 + 5000*(-b)), int(y0 + 5000*(a)) )
            pt2 = ( int(x0 - 5000*(-b)), int(y0 - 5000*(a) ))
            cv.line( img, pt1, pt2, red, width)


#Given image, find the track lines and return grouped lines (merged)
# Probabilistic - determines whether lines found using HoughLInes or HoughLinesP
#Returns: list of lines (lines as either YinterceptLine class or PolarLine depending)
def FindAndGroupLines(test_img, probabilistic = False):

    #Probabilistic determins if we use HoughLines or HoughLinesP
    #probabilistic = False

    #Current filter pipeline designed for 320x240, have to downsample mannually (future config camera)
    #test_img = cv.pyrDown(src=test_img)
    #img_gray = cv.cvtColor(test_img,cv.COLOR_BGR2GRAY)
    #img_gray = np.copy(test_img)
    img_gray = test_img

    #Step 1 - Pass through filters
    edges = FilterPipeline(img_gray)

    #Step 2 - Find all Lines within image
    lines = FindLines(edges,probabilistic)

    #Step 3 - Eliminate non-track lines, and then group to find most likely line
    #TODO: How to handle if camera FOV is large and sees more than 1 track lane? - bin by distance 2 center?
    best_lines = FilterForTrackLaneLines(lines, probabilistic)

    #convert to classes for ease of calculating x/y from line format
    prediction_lines = []
    if probabilistic:
        for i, line in enumerate(best_lines):
            prediction_lines.append(YInterceptLine(best_lines[i,0],best_lines[i,1]))
    else:
        for i, line in enumerate(best_lines):
            prediction_lines.append(PolarLine(best_lines[i,0],best_lines[i,1]))

    #draw lines on img for visualization
    for line in prediction_lines:
        line.draw(test_img)

    #Step 3.5 - Another elimination step. Check that all best_lines touch top and bottom of picture
    # --- this ended up filtering out some lines we want. may revisit later
    #prediction_lines = FilterForTopToBotLines(prediction_lines,test_img.shape)
    #prediction_lines = [line1]
    return prediction_lines

#Find the center point of track lane line in image    
#   Return as (x,y) - where y is assumed to be half the image height
def LaneCenterFinder(test_img, prediction_lines):

    #Step 4 - Find center
    #print("Lane Center Finder test_img shape", test_img.shape)
    middle_y = int(test_img.shape[0] / 2)
    #center_point = PredictLinesCenterX(middle_y,line1,line2)
    center_point = PredictLinesCenterX(middle_y,prediction_lines)

    #Draw output
    #For testing purposes
    cv.circle(test_img, center_point, 3, blue, thickness=3, lineType=8, shift=0)
    #DrawLines(test_img, lines,probabilistic)
    #cv.imwrite(output_folder + "line_img2.jpg",test_img)

    #Print output
    #CalculateScaledTrajectoryError(center_point, img_gray.shape)
    #print("Predicted center {}, Expected Center {}".format(center_point, (img_gray.shape[0]/2,img_gray.shape[1]/2) ))
    

    return center_point
    
def windows_mp4_video_writer(filename,frame_size,fps=24):
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    return cv.VideoWriter(filename,fourcc,fps,frame_size)

def play_video_with_line_detection(video_file,write_video=False):
    video_player = VideoPlayer.VideoPlayer(video_file)

    ret, next_frame = video_player.next_frame()
    quit = False
    if(write_video):
        w = 320
        h = 240
        video_out = windows_mp4_video_writer('processed_video.avi', (w,h), 26)
    while (ret == True) and (quit == False):
        ##PROCESS FRAME##
        #Current filter pipeline designed for 320x240, have to downsample mannually (future config camera)
        downsampled_orig = cv.pyrDown(src=next_frame)
        
        best_lines = FindAndGroupLines(downsampled_orig)
        LaneCenterFinder(downsampled_orig,best_lines)
        ##################

        #cv.imshow('frame',next_frame)
        quit = video_player.show_frame(downsampled_orig)
        ret, next_frame = video_player.next_frame()

        if write_video:
            video_out.write(downsampled_orig)

    if write_video:
        video_out.release()

if __name__ == "__main__":

    input_filename = 'InputImages/low_res_pic_14.jpg'
    img = cv.imread(input_filename)
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    timec1 = time.clock()
    timet1 = time.time()

    #Current filter pipeline designed for 320x240, have to downsample mannually stills (video camera configured)
    downsampled_orig = cv.pyrDown(src=img)
    img_gray = cv.pyrDown(src=img_gray, dst=img_gray ) #defaults to half size

    timec2 = time.clock()
    timet2 = time.time()
    #print(timet2-timet1,timec2-timec1)

    #Verify shape
    #downsampled_orig = img
    if ((downsampled_orig.shape[0]!=240) or  (downsampled_orig.shape[1] != 320) ):
        print("Error in expected shape. Got {} expexted {}".format(downsampled_orig.shape[0:2],(240,320)) )
        exit()
    #Test(img_gray)
    print ("downsampled_orig.shape",downsampled_orig.shape)
    best_lines = FindAndGroupLines(downsampled_orig)
    LaneCenterFinder(downsampled_orig,best_lines)

    print("---Video Playback---")
    #play_video_with_line_detection('InputImages/TrackDC_Intercepting_Lines.h264')
    play_video_with_line_detection('TrackDC_Video_3.h264')




