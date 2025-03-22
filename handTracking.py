import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color = (251,0,207), thickness = 2)
handConnectionsStyle = mpDraw.DrawingSpec(color = (0,255,0), thickness = 8)

while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result =  hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,handLmsStyle,handConnectionsStyle)
                for i, lm in enumerate(handLms.landmark):
                    xPosition =int( lm.x *imgWidth)
                    yPosition = int(lm.y * imgHeight)
                    print(i, xPosition, yPosition)

        cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
       break