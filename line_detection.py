import cv2 as cv
import numpy as np
import math as math

#Global Input Files (for testing)
input_filename = 'InputImages/trackpic14.jpg'
output_filename = 'OutputImages/output.jpg'
output_folder = 'OutputImages/'

img = cv.imread(input_filename)
cv.imwrite(output_filename,img)

#Canny edge detection
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imwrite(output_folder + "gray_image.jpg",img_gray)

def nothing(x):
    pass

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

    #strings 
    str_gauss_kernel_size = "gauss_kernel_size"
    str_gauss_sigma = "gauss_sigma"
    str_lower_canny_thresh = "lower_canny_thresh"
    str_upper_canny_thresh = "upper_canny_thresh"
    str_canny_kernel_size = "canny_kernel_size"
    
    cv.createTrackbar(str_gauss_kernel_size,window_name,gauss_kernel_size,50,nothing)
    cv.createTrackbar(str_gauss_sigma,window_name,gauss_sigma,100,nothing)
    cv.createTrackbar(str_lower_canny_thresh,window_name,lower_canny_thresh,255,nothing)
    cv.createTrackbar(str_upper_canny_thresh,window_name,upper_canny_thresh,255,nothing)
    cv.createTrackbar(str_canny_kernel_size,window_name,canny_kernel_size,50,nothing)
   
    #find center of shape and work on small sectin in center
    rows = img_gray.shape[0]
    cols = img_gray.shape[1]
    center_row = int(rows/2)
    center_col = int(cols/2)
    w_length = 350
    print("Rows/Cols ({},{}), Center Row/Col ({},{})".format(rows,cols,center_row,center_col))
    img_window = img_gray[center_row-w_length:center_row+w_length,center_col-w_length:center_col+w_length]
    
    while(1):
        #cv.imshow(window_name,
        cv.waitKey(1000)
        #print("gauss Kernal size {}".format(gauss_kernel_size))
        gauss_kernel_size = cv.getTrackbarPos(str_gauss_kernel_size,window_name)
        if (gauss_kernel_size % 2)==0:
            gauss_kernel_size += 1    
        gauss_sigma = cv.getTrackbarPos(str_gauss_sigma,window_name) / 10
        lower_canny_thresh = cv.getTrackbarPos(str_lower_canny_thresh,window_name) 
        upper_canny_thresh = cv.getTrackbarPos(str_upper_canny_thresh,window_name) 
        canny_kernel_size = cv.getTrackbarPos(str_canny_kernel_size,window_name) 
        if (canny_kernel_size % 2)==0:
            canny_kernel_size += 1
        gauss_kernel = (gauss_kernel_size,gauss_kernel_size)
        canny_kernel = (canny_kernel_size,canny_kernel_size)
        
        #gauss_img = cv.GaussianBlur(img_window,gauss_kernel, gauss_sigma, gauss_sigma,cv.BORDER_REPLICATE)
        gauss_img = cv.medianBlur(img_window,gauss_kernel_size)
        #gauss_img = cv.blur(img_window,gauss_kernel)
        edge_img = cv.Canny(gauss_img,lower_canny_thresh,upper_canny_thresh,canny_kernel_size)
        #stack images so they can be displayed in one window
        horizontal_stack = np.hstack((gauss_img,edge_img)) 
        cv.imshow(window_name,horizontal_stack)
        

#End RunCannyTuningWindow


def CannyFilter(img_gray):
    kernel_size = 15
    lower_canny_threshold = 50
    upper_canny_threshold = 100
    #gaussian filter
    #gauss_img = cv.GaussianBlur(img_gray,(5,5),0,0) #sigmaX/Y as 0's lets func determine
    filtered_img = cv.medianBlur(img_gray,kernel_size)
    cv.imwrite(output_folder + "median_filter.jpg",filtered_img)

    relative_image_max = np.max(img_gray)
    relative_image_min = np.min(img_gray)
    print("Max of gray image {}, Min {}".format(relative_image_max,relative_image_min) )
    #lower_canny_threshold = int(0.25 * relative_image_max)
    #upper_canny_threshold = int(0.5 * relative_image_max)
    print("Upper/Lower max: {} {}".format(lower_canny_threshold,upper_canny_threshold) )
    #canny_aperture = 11
    edges = cv.Canny(filtered_img,lower_canny_threshold,upper_canny_threshold)
    cv.imwrite(output_folder + "edges.jpg",edges)

    return edges

def FindLines(edge_img):
    rho_resolution = 1 #distance resultion of accumulator in pixels
    theta_resolution = np.pi / 180 #angle resulotion of accumulator in rads
    threshold = 200 #only return lines with greater number of votes

    #for HoughLInesP
    min_line_length = 100
    max_line_gap = 100

    #lines = cv.HoughLinesP(edge_img, rho=rho_resolution, theta=theta_resolution, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    lines = cv.HoughLines(edge_img,rho_resolution,theta_resolution,threshold)    

    return lines

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
    blue = [255,0,0]
    print("line1_point")
    print(line1_point)

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

    #kmeans clustering
    criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 0.001)
    clustering_lines = np.float32(clustering_lines)
    ret,label,centers=cv.kmeans(clustering_lines,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    print("KMeans Centers -- \n{}".format(centers))
    print(img.shape)

    #get line1 and line2 and calculate mid-points assuming center x
    line1 = centers[0]
    line2 = centers[1]
    middle_x = int(img.shape[1] / 2)
    line1_point = (middle_x, int(line1[0]*middle_x + line1[1]))
    line2_point = (middle_x, int(line2[0]*middle_x + line2[1]))
    blue = [255,0,0]
    print("line1_point")
    print(line1_point)

    cv.circle(img, line1_point, width, blue, thickness=3, lineType=8, shift=0)
    cv.circle(img, line2_point, width, blue, thickness=3, lineType=8, shift=0)

if __name__ == "__main__":
    #RunCannyTuningWindow(img_gray)
    edges = CannyFilter(img_gray)
    
    #Line detection and drawing
    line_img = np.copy(img)
    lines = FindLines(edges)
    #DrawLines(line_img,lines)
    DrawLinesThetaRho(line_img,lines)
    cv.imwrite(output_folder + "line_img.jpg",line_img)
