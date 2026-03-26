import cv2
import csv
from detector_maos import DetectorMaos

LETRA = "S"  # MUDE PARA B, C depois
ARQUIVO = f"dataset_{LETRA}.csv"

detector = DetectorMaos()
cap = cv2.VideoCapture(0)

with open(ARQUIVO, "w", newline="") as f:
    writer = csv.writer(f)

    print(f"Coletando letra {LETRA}... pressione Q para sair")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.encontrar_maos(frame)

        if detector.resultado and detector.resultado.multi_hand_landmarks:
            for mao in detector.resultado.multi_hand_landmarks:
                linha = []
                for p in mao.landmark:
                    linha.extend([p.x, p.y])
                writer.writerow(linha)
                print("Amostra salva")

        cv2.imshow("Coleta", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()