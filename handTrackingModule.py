import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detactionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detactionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=float(self.detectionCon),
            min_tracking_confidence=float(self.trackCon)
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]
        self.handLmsStyle =  self.mpDraw.DrawingSpec(color = (251,0,207), thickness = 2)
        self.handConnectionsStyle =  self.mpDraw.DrawingSpec(color = (0,255,0), thickness = 8)

    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,self.handLmsStyle,self.handConnectionsStyle)
        return img

    def findPosition(self, img, handNumber=0, draw=True):
        self.landmarkList = []
        xList = []
        yList = []
        bbox = []  # Initialize bbox

        imgHeight, imgWidth = img.shape[:2]  # Get image dimensions

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]

            # Collect all landmarks
            for i, lm in enumerate(myHand.landmark):
                xPosition = int(lm.x * imgWidth)
                yPosition = int(lm.y * imgHeight)
                xList.append(xPosition)
                yList.append(yPosition)

                self.landmarkList.append([i, xPosition, yPosition])

                # Draw a circle on the thumb tip (index 4)
                if i == 4:
                    cv2.circle(img, (xPosition, yPosition), 10, (0, 0, 255), cv2.FILLED)

            # Now calculate the bounding box AFTER collecting all points
            if xList and yList:
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = (xmin, ymin, xmax, ymax)

                # Draw the bounding box
                if draw:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        return self.landmarkList, bbox
    def fingersUp(self):
        fingers = []
        #thumb
        if self.landmarkList[self.tipIds[0]][1] > self.landmarkList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #fingers
        for id in range(1, 5):
            if self.landmarkList[self.tipIds[id]][2] < self.landmarkList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #totalFingers = fingers.count(1)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.landmarkList[p1][1:]
        x2, y2 = self.landmarkList[p2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2),(255,0,255),t)
            cv2.circle(img,(x1,y1), r, (255,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2), r, (255,0,255),cv2.FILLED)
            cv2.circle(img,(cx,cy), r, (0,0,255),cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
        
    
def main():   
    # create video object
    cap = cv2.VideoCapture(0)
    previousTime = 0
    currentTime = 0
    detector = handDetector()
    while True:
        success, img = cap.read()
        # if success:
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        # if len(landmarkList) != 0:
        #     print(landmarkList[4])

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img,F"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        cv2.imshow('img',img)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__=="__main__":
    main()