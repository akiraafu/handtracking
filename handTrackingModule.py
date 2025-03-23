import cv2
import mediapipe as mp
import time

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
    
    def findPosition(self,img, handNumber=0, draw=True):
        landmarkList = []
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            #check index number
            for i, lm in enumerate(myHand.landmark):
                xPosition =int( lm.x *imgWidth)
                yPosition = int(lm.y * imgHeight)
                # cv2.putText(img, str(i),(xPosition-25,yPosition+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),2)
                landmarkList.append([i,xPosition,yPosition])
                if i == 4:
                    cv2.circle(img,(xPosition,yPosition), 10,(0,0,255),cv2.FILLED)
        return landmarkList
        
    
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
        if len(landmarkList) !=0:
            print(landmarkList[4])

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img,F"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        cv2.imshow('img',img)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__=="__main__":
    main()