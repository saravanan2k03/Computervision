import cv2 as cv
import numpy as np
import matplotlib as plt
img = cv.imread('sudoku.png',0)
th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,11)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

cv.imshow("Image",img)
cv.imshow("ADAPTIVE_THRESH_MEAN_C",th2)
cv.imshow("ADAPTIVE_THRESH_GAUSSIAN_C",th3)

cv.waitKey(0) & 0xFF == ord('s')

cv.destroyAllWindows()