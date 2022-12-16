import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('opencv-logo.png')
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

titles = ['image','original']
images=[img,img1]

for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()