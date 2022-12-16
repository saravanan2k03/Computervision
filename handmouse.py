import cv2
import mediapipe as mp
import pyautogui

path = 'hand.jpg'
cap = cv2.VideoCapture(1)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width,screen_height = pyautogui.size()
index_y = 0
cap.set(3,1208)
cap.set(4,720)
while True:
    ret, frame = cap.read()
    # frame=cv2.imread(path)
    frame = cv2.flip(frame,1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame,hand)
            # print(hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
               x = int(landmark.x*frame_width)
               y = int(landmark.y*frame_height)
               # print(x,y)
               if id == 8:
                   cv2.circle(img=frame,center=(x,y),radius=10,color=(0,246,200))
                   index_x = screen_width/frame_width*x
                   index_y = screen_height/frame_height*y
                   pyautogui.moveTo(index_x,index_y)
               if id == 4:
                   cv2.circle(img=frame,center=(x,y),radius=10,color=(0,246,150))
                   thum_x = screen_width/frame_width*x
                   thum_y = screen_height/frame_height*y
                   # pyautogui.moveTo(thum_x,thum_y)
                   print("value",abs(index_y-thum_y))
                   print("not clicked")
                   if abs(index_y-thum_y)<20:
                       print("clicked")
                       pyautogui.click()
                       pyautogui.sleep(1)
    cv2.imshow("Virtual hands", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break