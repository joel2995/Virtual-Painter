import cv2
import mediapipe as mp
from time import sleep
from pynput import keyboard
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)  # Set width
cap.set(4, 7200)  # Set height

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

keyboard_ctrl = Controller()

keys = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]
]

finalText = ""

class Button():
    def _init_(self, pos, text, size=[60, 60]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([60 * j, 60 * i], key))

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_px = (int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0]))
                distance = abs(index_px[0] - button.pos[0]) + abs(index_px[1] - button.pos[1])

                if x < index_px[0] < x + w and y < index_px[1] < y + h:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green borders
                    cv2.putText(img, button.text, (x + 5, y + 40),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                    if distance < 30:
                        keyboard_ctrl.press(button.text)
                        finalText += button.text
                        sleep(0.25)

    # Draw keys with green borders
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green borders
        cv2.putText(img, button.text, (x + 5, y + 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Display pressed keys
    cv2.rectangle(img, (10, 550), (590, 690), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (20, 640),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()