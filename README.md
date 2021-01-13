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



