
"""


@author: akerur
"""

import cv2
import numpy as np
import random

def main(filename):
    if(filename == ""): #captures image from camera
        cam = cv2.VideoCapture(0)
        b, image = cam.read()
        cv2.imwrite("E:\image.jpg", image)
        fname = "E:\image.jpg"
        if b:    # frame captured without any errors
            cv2.imshow("camtest",image)
            k = cv2.waitKey(0) #key operations
            if k == 27:
                cv2.destroyAllWindows()
            elif k == ord('w'):
                w(image)
            elif k == ord('g'):
                g(fname)
            elif k == ord('G'):
                G(fname)
            elif k == ord('c'):
                c(fname)
            elif k == ord('s'):
                s(fname)            
            elif k == ord('x'):
                x(fname)
            elif k == ord('y'):
                y(fname)
            elif k == ord('m'):
                m(fname)
            elif k == ord('r'):
                r(fname)
            elif k == ord('h'):
                h()
            
         
    else: #loads image from specified path
       image = cv2.imread(filename)
       cv2.imshow('window',image)
       k = cv2.waitKey(0) #key operations
       if k == 27:
            cv2.destroyAllWindows()
       elif k == ord('w'):
           w(image)
       elif k == ord('g'):
           g(filename)
       elif k == ord('G'):
          G(filename)
       elif k == ord('c'):
           c(filename)
       elif k == ord('s'):
           s(filename)            
       elif k == ord('x'):
           x(filename)
       elif k == ord('y'):
           y(filename)
       elif k == ord('m'):
           m(filename)
       elif k == ord('r'):
           r(filename)
       elif k == ord('h'):
           h()

def i(filename): #refresh the previously processed image
    cv2.destroyAllWindows()
    main(filename)

def w(image): #saves processed image as out.jpg
    cv2.imwrite('out.jpg', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def g(filename): #creates a grayscale of the given image
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('GrayImage', gray)
    k = cv2.waitKey(0)
    if k == ord('w'):
        w(gray)
    if k == ord('i'):
        i(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def G(filename): #creates a grayscale of the given image using arithematic mean
    image = cv2.imread(filename)
    b,g,r = cv2.split(image)
    gray = (b+g+r/((b+g+r)/3)) #formula for calculating the arithematic mean
    cv2.imshow('MyGrayImage',gray)
    k = cv2.waitKey(0)
    if k == ord('w'):
        w(image)
    elif k == ord('i'):
        i(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()

def c(filename):
    image = cv2.imread(filename)
    b,g,r = cv2.split(image) #splitting the color channels
    image[:,:,random.randrange(0, 3, 1)] = 0 #manipulating the split b, g and r channels
    cv2.imshow('Color',image)
    k = cv2.waitKey(0)
    if k == ord('w'):
        w(image)
    elif k == ord('i'):
        i(filename)
    elif k == ord('c'):
        c(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()  

def nothing(x):
    pass
    
def s(filename): #smoothen the image with trackbar 
    image = cv2.imread(filename)
    cv2.namedWindow("smooth")
    cv2.createTrackbar("S", "smooth", 0, 100, nothing) #creates a trackbar
    while(1):
        pos = cv2.getTrackbarPos("S", "smoothen") #gets the trackbar position
        blurred = cv2.blur(image,(pos,pos))
        cv2.imshow("smooth", blurred)
    k = cv2.waitKey()
    if k == ord('w'):
        w(image)
    if k == ord('i'):
        i(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()

def x(filename): #performs a convolution of x derivative
    image = cv2.imread(filename)    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    xconvul = cv2.flip(grayscale, 0)
    cv2.imshow("Convolution", xconvul)
    k = cv2.waitKey(0)
    if k == ord('w'):
        w(xconvul)
    if k == ord('i'):
        i(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def y(filename): #performs a convolution of y derivative
    image = cv2.imread(filename)    
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yconvul = cv2.flip(gray_scale, 1)
    cv2.imshow("Convolution", yconvul)
    k = cv2.waitKey(0)
    if k == ord('w'):
        w(yconvul)
    if k == ord('i'):
        i(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def r(filename): #rotates the image on an angle Q
    image = cv2.imread(filename)
    row,col = image.shape[:2]
    center = (row / 2, col / 2)
    rot_mat = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (row , col))
    cv2.imshow("rotated", rotated)
    k = cv2.waitKey()
    if k == ord('w'):
        w(rotated)
    if k == ord('i'):
        i(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()

def m(filename): # Magnitude of a gradient based on x and y derivative
    image = cv2.imread(filename)
    xgrad = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=7) #calculates the sobel derivative of x
    ygrad = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)#calculates the sobel derivative of y
    cv2.convertScaleAbs(xgrad,xgrad) #normalized to range [0,255]
    cv2.convertScaleAbs(ygrad,ygrad)    
    mag = cv2.magnitude(xgrad,ygrad)
    cv2.imshow('Magnitude',mag)
    k = cv2.waitKey(0)
    if k == ord('w'):
        w(mag)
    elif k == ord('i'):
        i(filename)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

def h(): #help function
    print """
    This program performs different key operations on image. the image can be from a specified path or a camera capture.
    
    Key Operations: 
    
    'i’ - reload the original image (i.e. cancel any previous processing)
    ’w’ - save the current (possibly processed) image into the file ’out.jpg’
    ’g’ - convert the image to grayscale using the openCV conversion function.
    ’G’ - convert the image to grayscale using your implementation of conversion function.
    ’c’ - cycle through the color channels of the image showing a different channel every time the key is pressed.
    ’s’ - convert the image to grayscale and smooth it using the openCV function. Use a track bar to control the amount of smoothing.
    ’S’ - convert the image to grayscale and smooth it using your function which should perform convolution with a suitable filter. Use a track bar to control the amount of smoothing.
    ’x’ - convert the image to grayscale and perform convolution with an x derivative filter. Normalize the obtained values to the range [0,255].
    ’y’ - convert the image to grayscale and perform convolution with a y derivative filter. Normalize the obtained values to the range [0,255].
    ’m’ - show the magnitude of the gradient normalized to the range [0,255]. The gradient is computed based on the x and y derivatives of the image.
    ’p’ - convert the image to grayscale and plot the gradient vectors of the image every N pixels and let the plotted gradient vectors have a length of K. Use a track bar to control N. Plot the vectors as short line segments of length K.
    ’r’ - convert the image to grayscale and rotate it using an angle of Q degrees. Use a track bar to control the rotation angle. The rotation of the image should be performed using an inverse map so there are no holes in it.
    For further imformation on executing the program, please refer to the report under the folder Report."""
    cv2.destroyAllWindows()



    