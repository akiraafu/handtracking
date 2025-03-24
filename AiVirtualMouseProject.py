import cv2
import mediapipe as mp
import time
import handTrackingModule as htm
import numpy as np
import pyautogui

##########################
wCam, hCam = 640, 480
##########################
# create video object
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

previousTime = 0
currentTime = 0
detector = htm.handDetector(maxHands=1)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList, bbox = detector.findPosition(img)

    # 2. get the tip of the index and middle fingers
    if len(landmarkList) != 0:
        x1,y1 = landmarkList[8][1:]
        x2, y2 = landmarkList[12][1:]

        # 3. check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        # 4. Only Index Finger: Moving Mode
        # 5. Convert coordinates
        # 6. Smoothen Values
        # 7. Move Mouse
        # 8. Both Index and Middle fingers are up: Clicking mode
        # 9. Find distance between fingers
        # 10. Click mouse if distance short

    # 11. Frame Rate
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, F"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # 12 Display
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
       break
