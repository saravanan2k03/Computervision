import cv2
import numpy as np

def getcon(img,cThr=[100,100],showcCanny=False,minArea=1000,filter=0):
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(imggray,(5,5),1)
    imgCanny = cv2.Canny(imgblur,cThr[0],cThr[1])
    kernal = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernal,iterations=2)
    imgThre=cv2.erode(imgDial,kernal,iterations=2)
    if showcCanny:cv2.imshow('canny',imgThre)

    contours,hiearchy= cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalCountours=[]
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri=cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            bbox = cv2.boundingRect(approx)

            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append(len(approx),area,approx,bbox,i)

                else:
                    finalCountours.append(len(approx), area, approx, bbox, i)
    finalCountours =sorted(finalCountours,key= lambda x:x[1],reverse= True )

    # if draw:
    #     for con in  finalCountours:
    #         cv2.drawContours(img,con[4],-1,(0,0,255))