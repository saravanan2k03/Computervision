import numpy as np
import cv2

img = cv2.imread('opencv-logo.png')
img2 = cv2.imread('messi5.jpg')
print(img.shape)
print(img.size)
print(img.dtype)
b,g,r = cv2.split(img)
img = cv2.merge((r,g,b))
ball = img[280:340,330:390]
img[273:333,100:160] = ball
img =cv2.resize(img,(512,512))#resize function is decrease the image
img2 =cv2.resize(img2,(512,512))
# dst=cv2.add(img,img2)#add function in assign the image in stack
dst=cv2.addWeighted(img,.1,img2,.9,0)#addWeighted function is assign th image in stack with opacity
cv2.imshow('image',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()