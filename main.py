import cv2
import os
import threading
import winsound

from detector_maos import DetectorMaos
from classificador import classificar

audio_tocado = False
letra_estavel = None
contador = 0
FRAMES_CONFIRMAR = 8

# ===== CONTROLE DE TELA CHEIA =====
fullscreen = True
WINDOW_NAME = "Reconhecimento Libras"


def tocar_audio(letra):
    caminho = os.path.join("audios", f"letra{letra}.wav")
    if os.path.exists(caminho):
        winsound.PlaySound(caminho, winsound.SND_ASYNC)


def desenhar_interface(frame, letra=None, contador=0):
    h, w, _ = frame.shape

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (25, 25, 25), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(
        frame,
        "Reconhecimento de Libras",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2
    )

    if letra:
        status = "Letra confirmada"
        cor = (0, 200, 0)
    else:
        status = "Detectando gesto..."
        cor = (200, 200, 200)

    cv2.putText(
        frame,
        status,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        cor,
        2
    )

    if letra:
        cv2.rectangle(frame, (w - 260, 20), (w - 20, 110), (0, 180, 0), -1)
        cv2.putText(
            frame,
            f"LETRA {letra}",
            (w - 240, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (255, 255, 255),
            4
        )

    barra_max = 200
    progresso = min(contador / FRAMES_CONFIRMAR, 1.0)
    largura = int(barra_max * progresso)

    cv2.rectangle(frame, (20, 100), (20 + barra_max, 115), (80, 80, 80), -1)
    cv2.rectangle(frame, (20, 100), (20 + largura, 115), (0, 200, 0), -1)

    cv2.putText(
        frame,
        "Estabilidade",
        (230, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )

    cv2.putText(
        frame,
        "Q: sair | F: tela cheia",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1
    )

    return frame


def alternar_tela_cheia():
    global fullscreen
    if fullscreen:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL
        )
        fullscreen = False
    else:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )
        fullscreen = True


def main():
    global audio_tocado, letra_estavel, contador

    cap = cv2.VideoCapture(0)
    detector = DetectorMaos()

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.encontrar_maos(frame)

        letra_detectada = None

        if detector.resultado and detector.resultado.multi_hand_landmarks:
            for mao in detector.resultado.multi_hand_landmarks:
                letra_detectada = classificar(mao.landmark)

        if letra_detectada == letra_estavel and letra_detectada is not None:
            contador += 1
        else:
            letra_estavel = letra_detectada
            contador = 1

        letra_confirmada = None
        if contador >= FRAMES_CONFIRMAR and letra_estavel:
            letra_confirmada = letra_estavel

            if not audio_tocado:
                threading.Thread(
                    target=tocar_audio,
                    args=(letra_estavel,),
                    daemon=True
                ).start()
                audio_tocado = True
        else:
            audio_tocado = False

        frame = desenhar_interface(frame, letra_confirmada, contador)
        cv2.imshow(WINDOW_NAME, frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break
        elif tecla == ord("f"):
            alternar_tela_cheia()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()