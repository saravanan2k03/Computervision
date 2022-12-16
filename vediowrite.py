import cv2
cap = cv2.VideoCapture(0)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc,12.0,(640,480))
print(cap.isOpened())
cap.set(3,1208)
cap.set(4,720)
while (cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:
       print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
       print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       # out.write(frame)
       font = cv2.FONT_HERSHEY_SIMPLEX
       text = 'Width' + str(cap.get(3)) + 'Height' + str(cap.get(4))
       frame= cv2.putText(frame, text,(10,50),font,1,(0,255,255),2,cv2.LINE_AA)
       cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()



