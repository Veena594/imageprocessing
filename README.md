# imageprocessing
Program 1.Develop a program to display grayscale image using read and write operation.
grayscale image:Grayscale is a range of monochromatic shades from black to white.
Therefore,a grayscale image contains only shades of gray and no color.
to save image:cv2.imwrite()
to show image:cv2.imshow()
destroy all windows:cv2.destroyAllWindows()
import cv2
image=cv2.imread('tree.jpg')
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
cv2.imwrite('tree.jpg',image)
cv2.imshow("frame1",image)
cv2.imshow("frame2",grey_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Output:
![image](https://user-images.githubusercontent.com/72430475/104425483-949ee100-5535-11eb-8f7f-0b15e6080524.png)
![image](https://user-images.githubusercontent.com/72430475/104425732-dfb8f400-5535-11eb-9779-f648d9a0b838.png)

Program 2.Develop a program to perform linear transformation on image.
Linear transformation is a type of gray level transformation that is used for image enhancement.
It is a spatial domain method.
It is used for manipulation of an image so that result is more suitable than original for a specific appliaction.
Image scaling is the process of resizing a digital image.
Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms.
import cv2
import numpy as np
FILE_NAME = 'ip images1.png'
try: 
    img = cv2.imread(FILE_NAME) 
   (height, width) = img.shape[:2] 
    res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC) 
    cv2.imwrite('result.jpg', res) 
    cv2.imshow('image',img)
    cv2.imshow('result',res)
    cv2.waitKey(0)
  
except IOError: 
    print ('Error while reading files !!!')
    cv2.waitKey(0)
    cv2.destroyAllWindows(0)

Output:

![image](https://user-images.githubusercontent.com/72430475/104427156-c6b14280-5537-11eb-80be-3fff0cdd5c75.png)

import cv2 
import numpy as np 
  
FILE_NAME = 'ip images1.png'
img = cv2.imread(FILE_NAME) 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('result.jpg', res) 
cv2.waitKey(0)

Output:
![image](https://user-images.githubusercontent.com/72430475/104427420-1132bf00-5538-11eb-8ad5-253431eb0782.png)

Program 4.Write a program to convert color image into gray scale and binary image.

import cv2
img = cv2.imread("cat.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Binary Image",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

Output:


