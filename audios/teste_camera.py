import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
maos = mp.solutions.hands.Hands()
desenho = mp.solutions.drawing_utils

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = maos.process(rgb)

    if resultado.multi_hand_landmarks:
        for mao in resultado.multi_hand_landmarks:
            desenho.draw_landmarks(img, mao, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Teste Maos", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()