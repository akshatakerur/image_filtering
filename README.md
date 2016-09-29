# image_filtering

The program uses an image file either by reading it from a file or captured directly from the camera. 
The program has key functions that perform particular operations on the image when that particular key is pressed. 

##Key Functions

Clicking these keys on the keyboard modifies the displayed image accordingly:

’i’ - reloads the original image (i.e. cancel any previous processing)
’w’ - saves the current (possibly processed) image into the file ’out.jpg’
’g’ - converts the image to grayscale using the openCV conversion function.
'G' - creates a grayscale of the given image using arithematic mean
’c’ - cycles through the color channels of the image showing a different channel every time the key is pressed.
’s’ - converts the image to grayscale and smooth it using the openCV function. Use a track bar to control the amount of smoothing.
’x’ - converts the image to grayscale and perform convolution with an x derivative filter. Normalize the obtained values to the range [0,255].
’y’ - converts the image to grayscale and perform convolution with a y derivative filter. Normalize the obtained values to the range [0,255].
’m’ - shows the magnitude of the gradient normalized to the range [0,255]. The gradient is computed based on the x and y derivatives of the image.
’r’ - converts the image to grayscale and rotate it using an angle of Q degrees.
’h’ - Displays a short description of the program, its command line arguments, and the keys it supports.
