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

Program 3.Develop a program to find the sum and mean of a set of images.Create n number of images and read from the directory and perform the operations.
Mean:mean value gives the contributions of individual pixel intensity for the entire image.
sum:adds the value of each pixel in one of the input iamges with the corrresponding pixel.
in the other input image and returns the sum in the corresponding pixel of the output image.
import cv2
import os
path = "D:\imp_for_ip"
imgs=[]
dirs=os.listdir(path)

for file in dirs:
    fpat=path+"\\"+file
    imgs.append(cv2.imread(fpat))
    
i=0
sum_img=[]
for sum_img in imgs:
    read_imgs=imgs[i]
    sum_img=sum_img+read_imgs
    #cv2.imshow(dirs[i],imgs[i])
    i=i+1
print(i)
cv2.imshow('sum',sum_img)
print(sum_img)

cv2.imshow('mean',sum_img/i)
mean=(sum_img/i)
print(mean)

cv2.waitKey()
cv2.destroyAllwindows()

Output:
![image](https://user-images.githubusercontent.com/72430475/104433318-d2ecce00-553e-11eb-8231-abcc2eade4fc.png)


Program 4.Write a program to convert color image into gray scale and binary image.
A method called threshold() is used to convert grayscale images to binary image.
Binary image:Binary image is the type of image where each pixel is black or white/0 or 1.Here 0 represents black and 1 represents white pixel.
Grayscale image:Grayscale is a range of monochromatic shades from black to white.
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
![image](https://user-images.githubusercontent.com/72430475/104431592-e9922580-553c-11eb-8070-9e6160d0ace0.png)
![image](https://user-images.githubusercontent.com/72430475/104431731-15ada680-553d-11eb-983e-7cc8f27ae326.png)

Program 5.Write a program to convert color image into different color space.

import cv2
img = cv2.imread("cat.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
cv2.imshow("GRAY image",gray)
cv2.waitKey(0)
cv2.imshow("HSV image",hsv)
cv2.waitKey(0)
cv2.imshow("LAB image",lab)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.destroyAllWindows()

Output:
![image](https://user-images.githubusercontent.com/72430475/104431990-64f3d700-553d-11eb-9021-67eba2e74158.png)
![image](https://user-images.githubusercontent.com/72430475/104432199-9f5d7400-553d-11eb-9ee5-5f39bbc59ba9.png)
![image](https://user-images.githubusercontent.com/72430475/104432346-c9169b00-553d-11eb-845c-3b2f0fe0ceca.png)

Program 6.Develop a program to create an image from 2D array.

import numpy as np
from PIL import Image
import cv2 as c 
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [150, 128, 0] #Orange left side
array[:,100:] = [0, 0, 255]   #Blue right side

img = Image.fromarray(array)
img.save('img1.png')
img.show()
c.waitKey(0)

Output:
![image](https://user-images.githubusercontent.com/72430475/104432656-1430ae00-553e-11eb-8c2d-be454cafefcf.png)


