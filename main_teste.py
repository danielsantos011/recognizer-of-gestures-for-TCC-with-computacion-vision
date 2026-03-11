# main_teste.py

import cv2
from detector_maos import DetectorMaos
from treino_gesto import letra_a

cap = cv2.VideoCapture(0)
detector = DetectorMaos()

contador_a = 0

while True:
    sucesso, imagem = cap.read()
    if not sucesso:
        break

    imagem = cv2.flip(imagem, 1)

    imagem = detector.encontrar_maos(imagem)

    if detector.resultado and detector.resultado.multi_hand_landmarks:
        landmarks = detector.resultado.multi_hand_landmarks[0].landmark

        if letra_a(landmarks):
            contador_a += 1
        else:
            contador_a = 0

        # Confirma após alguns frames
        if contador_a > 8:
            cv2.putText(
                imagem,
                "A",
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (0, 255, 0),
                6
            )

    cv2.imshow("Teste Libras - Letra A", imagem)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()