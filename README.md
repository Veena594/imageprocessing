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
