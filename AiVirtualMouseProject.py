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
detector = htm.handDetector()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    # 2. get the tip of the index and middle fingers
    # 3. check which fingers are up
    # 4. Only Index Finger: Moving Mode
    # 5. Convert coordinates
    # 6. Smoothen Values
    # 7. Move Mouse
    # 8. Both Index and Middle fingers are up: Clicking mode
    # 9. Find distance between fingers
    # 10. Click mouse if distance short
    # 11. Frame Rate
    # 12 Display
    if success:
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        # if len(landmarkList) !=0:
        #     print(landmarkList[4])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, F"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
                break
