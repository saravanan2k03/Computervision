import numpy as np
import cv2

# events = [i for i in dir(cv2) if 'EVENT' in i ]
# print(events)


def Click_event(event,x,y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,' , ',y)
        strxy = str(x) + ','+str(y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #reference first in image second in text third is location fourth is font fivth is scale sixth is color last is thicknes
        cv2.putText(img,strxy,(x,y),font,.5,(255,255,0),2)
        cv2.imshow('image',img)
    if event == cv2.EVENT_RBUTTONDOWN:
       blue = img[y,x,0]
       green = img[y,x,1]
       red = img[y,x,2]
       strbgr = str(blue) + ',' + str(green) + ',' + str(red)
       font = cv2.FONT_HERSHEY_SIMPLEX
       cv2.putText(img, strbgr, (x, y), font, .5, (255, 200, 255), 2)
       cv2.imshow('image', img)
# img = np.zeros((512,700,3),np.uint8)
img = cv2.imread('cards.jpg', 1)
cv2.imshow('image',img)
cv2.setMouseCallback('image',Click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()