import mediapipe as mp
import cv2

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            selectedHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(selectedHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw and id == 8:  # Example: Draw circle on the tip of the index finger
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

    def fingersUp(self):
        fingers = []
        if not self.results or not self.results.multi_hand_landmarks:
            return fingers
        # Thumb
        if self.lmList[4][1] > self.lmList[3][1]:  # Compare x-coordinates for thumb
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(8, 21, 4):  # Starts at 8 (index finger) and goes to 20 (pinky)
            if self.lmList[id][2] < self.lmList[id - 2][2]:  # Compare y-coordinates for fingers
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionCon=0.85)
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPositions(img)
        if len(lmList) != 0:
            print(lmList)
            fingers = detector.fingersUp()
            print(fingers)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
