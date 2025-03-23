import cv2
import mediapipe as mp
import time
import handTrackingModule as htm

 # create video object
cap = cv2.VideoCapture(0)
previousTime = 0
currentTime = 0
detector = htm.handDetector()
while True:
        success, img = cap.read()
        # if success:
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        if len(landmarkList) !=0:
            print(landmarkList[4])

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img,F"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        cv2.imshow('img',img)

        if cv2.waitKey(1) == ord('q'):
            break
