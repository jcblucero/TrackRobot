import cv2 as cv
import numpy as np

#Global Input Files (for testing)
input_filename = 'InputImages/trackpic3.jpg'
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

    lines = cv.HoughLinesP(edge_img, rho=rho_resolution, theta=theta_resolution, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    #lines = cv.HoughLines(edge_img,rho_resolution,theta_resolution,threshold)    

    return lines

def DrawLines(img):
    red = [0,0,255]
    width = 3
    
    if lines is None:
        print( "No lines found")
        return 

    print("Lines Shape:")
    print(lines.shape)
    count = 0
    for line in lines:
        print("Line #{}".format(count) )
        count += 1
        coords = line[0]
        pt1 = (coords[0],coords[1])
        pt2 = (coords[2],coords[3])
        cv.line(img, pt1, pt2, red, width)

if __name__ == "__main__":
    #RunCannyTuningWindow(img_gray)
    edges = CannyFilter(img_gray)
    
    #Line detection and drawing
    line_img = np.copy(img)
    lines = FindLines(edges)
    DrawLines(line_img)
    cv.imwrite(output_folder + "line_img.jpg",line_img)
