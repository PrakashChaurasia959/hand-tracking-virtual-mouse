import cv2
import mediapipe as mp

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera open nahi ho raha")
    exit()

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

draw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_id, hand in enumerate(result.multi_hand_landmarks):

            # draw hand skeleton
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            index = hand.landmark[8]

            x = int(index.x * w)
            y = int(index.y * h)

            # show tracking points
            if hand_id == 0:
                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)  # right hand

            if hand_id == 1:
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # left hand

    cv2.putText(frame, "2 HAND TRACKING MODE", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, "ESC = Exit", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()