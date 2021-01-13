# imageprocessing
Program 1.Develop a program to display grayscale image using read and write operation.
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
