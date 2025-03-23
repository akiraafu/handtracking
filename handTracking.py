import cv2
import mediapipe as mp
import time

# create video object
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color = (251,0,207), thickness = 2)
handConnectionsStyle = mpDraw.DrawingSpec(color = (0,255,0), thickness = 8)

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    if success:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result =  hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,handLmsStyle,handConnectionsStyle)
                #check index number
                for i, lm in enumerate(handLms.landmark):
                    xPosition =int( lm.x *imgWidth)
                    yPosition = int(lm.y * imgHeight)
                    # cv2.putText(img, str(i),(xPosition-25,yPosition+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),2)
                    if i == 4:
                        cv2.circle(img,(xPosition,yPosition), 10,(0,0,255),cv2.FILLED)
                    print(i, xPosition, yPosition)

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img,F"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

        cv2.imshow('img',img)

    if cv2.waitKey(1) == ord('q'):
       break