import cv2 as cv
import numpy as np

#Filter Tuning Window
#Input - Grayscale Image
#Creates a window with slider bars for tuning a canny edge detector, gaussian filter, and image dilation
#Window displays output canny image
def RunFilterTuningWindow(img_gray):
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
            src=gauss_img,ksize=gauss_kernel, sigmaX=gauss_sigma, sigmaY=gauss_sigma, \
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

if __name__ == "__main__":
    input_filename = 'InputImages/low_res_pic_14.jpg'
    img = cv.imread(input_filename)
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    RunFilterTuningWindow(img_gray)