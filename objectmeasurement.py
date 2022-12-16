import cv2
import numpy as np
import utils
#########################

webcam = False
path = 'cards.jpg'

cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)

while True:
   if webcam: sucess, img =cap.read()
   else:img= cv2.imread(path)
   utils.getcon(img,showcCanny=True)
   cv2.imshow('Original',img)
   if cv2.waitKey(1) & 0xFF == ord('s'):
     break