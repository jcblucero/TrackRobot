import cv2 as cv
import numpy as np
import math as math
import time

#Global Input Files (for testing)
input_filename = 'InputImages/low_res_pic_11.jpg'
output_filename = 'OutputImages/output.jpg'
output_folder = 'OutputImages/'

img = cv.imread(input_filename)
print("Image shape",img.shape)
cv.imwrite(output_filename,img)

#Colors for drawing on image
green = [0,255,0]
purple = [255,0,255]
blue = [255,0,0]

#Canny edge detection
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print("Gray Image Shape",img_gray.shape)
cv.imwrite(output_folder + "gray_image.jpg",img_gray)

def nothing(x):
    pass
#
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
#
###

#Canny Tuning Window
#Input - Grayscale Image
#Creates a window with slider bars for tuning a canny edge detector with gaussian filter. 
#Window displays output canny image
def RunCannyTuningWindow(img_gray):
    window_name = "Canny Edge Detector"
    cv.namedWindow(window_name)

    #parameters to tune
    gauss_kernel_size = 3    
    gauss_sigma = 10
    lower_canny_thresh = 100
    upper_canny_thresh = 200
    canny_kernel_size = 3
    dilate_size = 1

    #strings 
    str_gauss_kernel_size = "gauss_kernel_size"
    str_gauss_sigma = "gauss_sigma"
    str_lower_canny_thresh = "lower_canny_thresh"
    str_upper_canny_thresh = "upper_canny_thresh"
    str_canny_kernel_size = "canny_kernel_size"
    str_dilate_size = "dilate_size"
    
    cv.createTrackbar(str_gauss_kernel_size,window_name,gauss_kernel_size,50,nothing)
    cv.createTrackbar(str_gauss_sigma,window_name,gauss_sigma,100,nothing)
    cv.createTrackbar(str_lower_canny_thresh,window_name,lower_canny_thresh,400,nothing)
    cv.createTrackbar(str_upper_canny_thresh,window_name,upper_canny_thresh,400,nothing)
    cv.createTrackbar(str_canny_kernel_size,window_name,canny_kernel_size,50,nothing)
    cv.createTrackbar(str_dilate_size,window_name,dilate_size,20,nothing)
   
    #find center of shape and work on small sectin in center
    #Only do this for 1080/1920
    rows = img_gray.shape[0]
    cols = img_gray.shape[1]
    #Downsample to some size. Size input is COLS then ROWS
    #downsample_size = (128,96) apparently you can't downsample more than 1/2 size at a time...
    img_gray = cv.pyrDown(src=img_gray, dst=img_gray ) #defaults to half size
    #img_gray = cv.pyrDown(src=img_gray, dst=img_gray )

    #print("Rows/Cols ({},{}), Center Row/Col ({},{})".format(rows,cols,center_row,center_col))
    if(rows == 1080) and (cols == 1920):
        center_row = int(rows/2)
        center_col = int(cols/2)
        w_length = 350
        print("Rows/Cols ({},{}), Center Row/Col ({},{})".format(rows,cols,center_row,center_col))
        img_window = img_gray[center_row-w_length:center_row+w_length,center_col-w_length:center_col+w_length]
    else:
        print("Rows/Cols ({},{})".format(rows,cols))
        img_window = img_gray

    while(1):
        #cv.imshow(window_name,
        cv.waitKey(1000)
        #print("gauss Kernal size {}".format(gauss_kernel_size))
        gauss_kernel_size = cv.getTrackbarPos(str_gauss_kernel_size,window_name)
        dilate_size = cv.getTrackbarPos(str_dilate_size,window_name)
        if (gauss_kernel_size % 2)==0:
            gauss_kernel_size += 1    
        gauss_sigma = cv.getTrackbarPos(str_gauss_sigma,window_name) / 10.0
        lower_canny_thresh = cv.getTrackbarPos(str_lower_canny_thresh,window_name) 
        upper_canny_thresh = cv.getTrackbarPos(str_upper_canny_thresh,window_name) 
        canny_kernel_size = cv.getTrackbarPos(str_canny_kernel_size,window_name) 
        if (canny_kernel_size % 2)==0:
            canny_kernel_size += 1
        gauss_kernel = (gauss_kernel_size,gauss_kernel_size)
        canny_kernel = (canny_kernel_size,canny_kernel_size)

        #Dilate to get max within kernel
        gauss_img = img_window
        dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilate_size*2+1,dilate_size*2+1) )
        gauss_img = cv.dilate(src=gauss_img, kernel=dilate_kernel)
        
        gauss_img = cv.GaussianBlur(
            src=gauss_img,ksize=gauss_kernel, sigmaX=gauss_sigma, sigmaY=gauss_sigma,\
            borderType=cv.BORDER_REPLICATE)
        #gauss_img = cv.medianBlur(gauss_img,gauss_kernel_size)
        #gauss_img = cv.blur(
        #    src=gauss_img,ksize=gauss_kernel,borderType=cv.BORDER_REPLICATE)

        """
        #Dilate to get max within kernel
        #gauss_img = img_window
        dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (dilate_size*2+1,dilate_size*2+1) )
        gauss_img = cv.dilate(src=gauss_img, kernel=dilate_kernel)
        """
        edge_img = cv.Canny(
            image=gauss_img,threshold1=lower_canny_thresh,threshold2=upper_canny_thresh, \
            apertureSize=canny_kernel_size)
        #edge_img = cv.dilate(src=edge_img, kernel=dilate_kernel)
        #print(canny_kernel_size)
        #stack images so they can be displayed in one window
        horizontal_stack = np.hstack((gauss_img,edge_img)) 
        cv.imshow(window_name,horizontal_stack)       
#
#
##########End RunCannyTuningWindow


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
    cv.imwrite(output_folder + "filtered_image.jpg",filtered_img)

    #Step 3 of Filter - Canny Edge Detector
    #edges = cv.Canny(filtered_img,lower_canny_threshold,upper_canny_threshold)
    edges = cv.Canny(
        image=filtered_img,threshold1=lower_canny_threshold,threshold2=upper_canny_threshold, \
        apertureSize=canny_kernel_size)
    cv.imwrite(output_folder + "edges.jpg",edges)

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

#

# Take HoughLines and eliminate unlikely candidates. Then group using kmeans and output as np array
def FilterForTrackLaneLines(lines, probabilistic=False):

    #Step 1: Take output of HoughLines(P) and convert to useful np matrix format
    if probabilistic:
        clustering_lines = np.zeros( (lines.shape[0],2) )
        count = 0
        for line in lines:
            print("Line #{}".format(count) )
            coords = line[0]
            pt1 = (coords[0],coords[1])
            pt2 = (coords[2],coords[3])

            #Creating numpy matrix from tuple to feed to kmeans
            slope_intercept = ConvertToSlopeIntercept(coords)
            print("Slope = {} , B = {} ".format(slope_intercept[0],slope_intercept[1]))
            clustering_lines[count] = np.array(slope_intercept)
            count += 1

    else:
        matrix_shape = (lines.shape[0],lines.shape[2])
        print (lines.reshape(matrix_shape))
        clustering_lines = lines.reshape(matrix_shape)

    #Step 2 - Eliminate lines that are unlikely to be track lane lines
    if probabilistic:
        #Filter lines that are too horizontal (+-30deg of 0 which is horizontal)
        #tan(theta) = opp/adj -> rise/run -> slope. tan(theta) == slope
        lower_angle_threshold = np.tan(30 * np.pi/180)
        upper_angle_threshold = np.tan(-30 * np.pi/180)

        selection_array = (clustering_lines[:,0] >= lower_angle_threshold) | (clustering_lines[:,0] <= upper_angle_threshold)
        clustering_lines = clustering_lines[selection_array] 
        print ("filtered clustering lines")
        print (clustering_lines)
    else:
        #Filter lines that are too horizontal (60-120deg, +-30deg of 90 which is horizontal)
        lower_angle_threshold = 60 * np.pi/180
        upper_angle_threshold = 120 * np.pi/180

        selection_array = (clustering_lines[:,1] <= lower_angle_threshold) | (clustering_lines[:,1] >= upper_angle_threshold)
        clustering_lines = clustering_lines[selection_array] 
        print (clustering_lines)

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
        best_lines[:,1] = np.arctan(centers_xy[:,0]/centers_xy[:,1])

        print("KMeans Centers -- \n{}".format(best_lines))

    return best_lines

#Predict the x position corresponding to y for each line, then take the middle of those two
def PredictLinesCenterX(y,line1,line2):
    point1 = (line1.predict_x(y),y)
    point2 = (line2.predict_x(y),y)

    center_point = FindCenter(point1,point2)
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

def DrawLinesThetaRho(img,lines):
    red = [0,0,255]
    width = 3
    
    if lines is None:
        print( "No lines found")
        return 

    print("Lines Shape:")
    print(lines.shape)
    max_lines_to_print = 2
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

    #####
    # Xcos(theta) + Ysin(theta) = r
    matrix_shape = (lines.shape[0],lines.shape[2])
    print (lines.reshape(matrix_shape))
    clustering_lines = lines.reshape(matrix_shape)
    print ("clustering_Lines shape {}".format(clustering_lines.shape))

    #Filter lines that are too horizontal (60-120deg, +-30deg of 90 which is horizontal)
    lower_angle_threshold = 60 * np.pi/180
    upper_angle_threshold = 120 * np.pi/180

    selection_array = (clustering_lines[:,1] <= lower_angle_threshold) | (clustering_lines[:,1] >= upper_angle_threshold)
    clustering_lines = clustering_lines[selection_array] 
    print (clustering_lines)

    #Convert to x/y form using sin/cos to avoid discontinuities in (theta 0-2pi, and rho negative distance)
    xy_clustering = np.copy(clustering_lines)
    xy_clustering[:,0] = np.sin(clustering_lines[:,1]) * clustering_lines[:,0] #y's = sin(theta) * rho
    xy_clustering[:,1] = np.cos(clustering_lines[:,1]) * clustering_lines[:,0] #x's = cos(theta) * rho
    
    #kmeans clustering
    criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 0.001)
    xy_clustering = np.float32(xy_clustering)
    ret,label,centers_xy=cv.kmeans(xy_clustering,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    #convert back to theta/rho
    centers=np.copy(centers_xy)
    centers[:,0] = np.sqrt(centers_xy[:,0]*centers_xy[:,0] + centers_xy[:,1]*centers_xy[:,1]) #rho
    centers[:,1] = np.arctan(centers_xy[:,0]/centers_xy[:,1])

    print("KMeans Centers -- \n{}".format(centers))

    """
    #kmeans clustering
    criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 0.001)
    clustering_lines = np.float32(clustering_lines)
    ret,label,centers=cv.kmeans(clustering_lines,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    print("KMeans Centers -- \n{}".format(centers))
    
    #Translate Line to points -- Y = (r-Xcos(theta)) / sin(theta)
    
    line1 = centers[0]
    rho1,theta1 = (line1[0],line1[1])
    line2 = centers[1]
    rho2,theta2 = (line2[0],line2[1])
    
    middle_x = int(img.shape[1] / 2)
    line1_point = (middle_x, int( (rho1-(middle_x*math.cos(theta1)))/math.sin(theta1) ))
    line2_point = (middle_x, int( (rho2-(middle_x*math.cos(theta2)))/math.sin(theta2) ))
    """
    #Find x's instead of y's for vertical orientation
    #X = (r-Ysin(theta)) / cos(theta)
    line1 = centers[0]
    rho1,theta1 = (line1[0],line1[1])
    line2 = centers[1]
    rho2,theta2 = (line2[0],line2[1])

    middle_y = int(img.shape[0] / 2)
    line1_point = ( int( (rho1-(middle_y*math.sin(theta1)))/math.cos(theta1) ), middle_y)
    line2_point = ( int( (rho2-(middle_y*math.sin(theta2)))/math.cos(theta2) ), middle_y)

    center_point = FindCenter(line1_point,line2_point)
    cv.circle(img, center_point, width, purple, thickness=3, lineType=8, shift=0)

    CalculateScaledTrajectoryError(center_point, img.shape)

    blue = [255,0,0]
    print("line1_point")
    print(line1_point)

    print("Predicted center {}, Expected Center {}".format(center_point, (img.shape[0]/2,img.shape[1]/2) ))
    print(center_point)

    cv.circle(img, line1_point, width, blue, thickness=3, lineType=8, shift=0)
    cv.circle(img, line2_point, width, blue, thickness=3, lineType=8, shift=0)
        
    
def DrawLines(img,lines):
    red = [0,0,255]
    width = 3
    
    if lines is None:
        print( "No lines found")
        return 

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

        #Creating numpy matrix from tuple to feed to kmeans
        if count==1:
            clustering_lines = np.array( slope_intercept )
        else:
            clustering_lines = np.vstack((np.array(slope_intercept),clustering_lines))

        
        #if(count == max_lines_to_print):
        #    break
    ############################
    print(clustering_lines)
    
    #Filter lines that are too horizontal (+-30deg of 0 which is horizontal)
    #tan(theta) = opp/adj -> rise/run -> slope. tan(theta) == slope
    lower_angle_threshold = np.tan(30 * np.pi/180)
    upper_angle_threshold = np.tan(-30 * np.pi/180)

    selection_array = (clustering_lines[:,0] >= lower_angle_threshold) | (clustering_lines[:,0] <= upper_angle_threshold)
    clustering_lines = clustering_lines[selection_array] 
    print ("filtered clustering lines")
    print (clustering_lines)


    #kmeans clustering
    criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 0.001)
    clustering_lines = np.float32(clustering_lines)
    ret,label,centers=cv.kmeans(clustering_lines,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    print("KMeans Centers -- \n{}".format(centers))
    print(img.shape)

    #get line1 and line2 and calculate mid-points assuming center x
    line1 = centers[0]
    line2 = centers[1]
    #actually doing middle y (to get x) because orientation is vertical now
    """
    middle_x = int(img.shape[1] / 2)
    line1_point = (middle_x, int(line1[0]*middle_x + line1[1]))
    line2_point = (middle_x, int(line2[0]*middle_x + line2[1]))
    """
    #x = (y-b) / m
    middle_y = int(img.shape[0] / 2)
    line1_point = (int( (middle_y-line1[1])/line1[0]), middle_y)
    line2_point = (int( (middle_y-line2[1])/line2[0]), middle_y)
    blue = [255,0,0]
    print("line1_point")
    print(line1_point)

    cv.circle(img, line1_point, width, blue, thickness=3, lineType=8, shift=0)
    cv.circle(img, line2_point, width, blue, thickness=3, lineType=8, shift=0)

#Calculate the raw error which will eventually be used by PID controller for steering
#left/right steering only, so only 1d input
def CalculateRawError(measured, desired):
    return desired - measured

#Calculate the error which will be used to feed PID controller for steering
#We use a scaled error between [-100.0,100.0] (float) so that PID is impartial to viewing dimensions
#Inputs: center_point - (x,y) pair  - center of lines found through line detection
#       image_dimensions - tuple - dimensions of image fed by camera to line detection
#Outputs: Float between [-100.0,100.0] representing error. (by convention, negative means measured center is right of image center)
def CalculateScaledTrajectoryError( center_point, image_dimensions):
    
    print("Image Dimensions {}".format(image_dimensions))

    #We care about lateral error,since we are only steering left/right.
    #This means x direction (columns) of image
    measured_value = center_point[0] #x index is 0
    desired_value = int(image_dimensions[1] / 2) #TODO: check this index
    max_error = desired_value #TODO: check this index

    raw_error = float(desired_value - measured_value)

    scaled_error = (raw_error / max_error) * 100.0

    print("Max Error = {}, Raw Error = {}, Scaled Error = {}".format(max_error, raw_error,scaled_error) )

    return scaled_error
    
def Test(test_img):

    #Probabilistic determins if we use HoughLines or HoughLinesP
    probabilistic = True

    #Current filter pipeline designed for 320x240, have to downsample mannually (future config camera)
    #test_img = cv.pyrDown(src=test_img)
    img_gray = cv.cvtColor(test_img,cv.COLOR_BGR2GRAY)

    #Step 1 - Pass through filters
    edges = FilterPipeline(img_gray)

    #Step 2 - Find all Lines within image
    lines = FindLines(edges,probabilistic)

    #Step 3 - Eliminate non-track lines, and then group to find most likely line
    #TODO: How to handle if camera FOV is large and sees more than 1 track lane? - bin by distance 2 center?
    best_lines = FilterForTrackLaneLines(lines, probabilistic)
    #convert to classes for ease of calculating x/y from line format
    if probabilistic:
        line1 = YInterceptLine(best_lines[0,0],best_lines[0,1])
        line2 = YInterceptLine(best_lines[1,0],best_lines[1,1])        
    else:
        line1 = PolarLine(best_lines[0,0],best_lines[0,1])
        line2 = PolarLine(best_lines[1,0],best_lines[1,1])

    #Step 4 - Find center
    middle_y = int(img_gray.shape[0] / 2)
    center_point = PredictLinesCenterX(middle_y,line1,line2)

    #Draw output
    cv.circle(test_img, center_point, 3, blue, thickness=3, lineType=8, shift=0)
    cv.imwrite(output_folder + "line_img2.jpg",test_img)
    

if __name__ == "__main__":
    #RunCannyTuningWindow(img_gray)
    """#For 1920x1080 fullscale image
    edges = CannyFilter(img_gray)
    
    #Line detection and drawing
    line_img = np.copy(img)
    lines = FindLines(edges)
    #DrawLines(line_img,lines)
    DrawLinesThetaRho(line_img,lines)
    cv.imwrite(output_folder + "line_img.jpg",line_img)
    """

    timec1 = time.clock()
    timet1 = time.time()
    #timec1 = time.clock()

    #Current filter pipeline designed for 320x240, have to downsample mannually (future config camera)
    downsampled_orig = cv.pyrDown(src=img)
    img_gray = cv.pyrDown(src=img_gray, dst=img_gray ) #defaults to half size
    edges = FilterPipeline(img_gray)

    #Line detection and drawing
    line_img = np.copy(downsampled_orig)
    lines = FindLines(edges)
    #DrawLines(line_img,lines)
    DrawLinesThetaRho(line_img,lines)
    cv.imwrite(output_folder + "line_img.jpg",line_img)

    time.sleep(1)

    timec2 = time.clock()
    timet2 = time.time()
    #timec2 = time.clock()

    print(time.time(),time.clock())
    print(timet2-timet1,timec2-timec1)


    print("----------Testing Refactor--------")
    Test(downsampled_orig)




