import cv2
import os
import threading
import winsound

from detector_maos import DetectorMaos
from classificador import classificar

# ===============================
# VARIÁVEIS GLOBAIS
# ===============================
audio_tocado = False
letra_estavel = None
contador = 0
FRAMES_CONFIRMAR = 8

palavra_atual = ""
texto_final = ""


# ===============================
# FUNÇÃO DE ÁUDIO
# ===============================
def tocar_audio(letra):
    caminho = os.path.join("audios", f"letra{letra}.wav")
    if os.path.exists(caminho):
        winsound.PlaySound(caminho, winsound.SND_ASYNC)


# ===============================
# INTERFACE GRÁFICA
# ===============================
def desenhar_interface(frame, letra=None, contador=0, palavra="", texto=""):
    h, w, _ = frame.shape

    # Painel superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (25, 25, 25), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Título
    cv2.putText(
        frame,
        "Reconhecimento de Libras",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2
    )

    # Status
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

    # Caixa da letra
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

    # Barra de estabilidade
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

    # Texto digitado
    cv2.putText(
        frame,
        f"Digitando: {palavra}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Texto: {texto}",
        (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2
    )

    # Rodapé
    cv2.putText(
        frame,
        "B: enviar | N: apagar | Q: sair",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1
    )

    return frame


# ===============================
# FUNÇÃO PRINCIPAL
# ===============================
def main():
    global audio_tocado, letra_estavel, contador
    global palavra_atual, texto_final

    cap = cv2.VideoCapture(0)
    detector = DetectorMaos()

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

        # ===============================
        # CONTROLE DE ESTABILIDADE
        # ===============================
        if letra_detectada == letra_estavel and letra_detectada is not None:
            contador += 1
        else:
            letra_estavel = letra_detectada
            contador = 1

        # ===============================
        # CONFIRMAÇÃO DA LETRA
        # ===============================
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

        # Adiciona letra à palavra
        if letra_confirmada:
            if not palavra_atual.endswith(letra_confirmada):
                palavra_atual += letra_confirmada

        # ===============================
        # INTERFACE
        # ===============================
        frame = desenhar_interface(
            frame,
            letra_confirmada,
            contador,
            palavra_atual,
            texto_final
        )

        cv2.imshow("Reconhecimento Libras", frame)

        # ===============================
        # TECLAS
        # ===============================
        tecla = cv2.waitKey(1) & 0xFF

        # Enviar palavra (B)
        if tecla == ord("b") or tecla == ord("B"):
            if palavra_atual:
                texto_final += palavra_atual + " "
                palavra_atual = ""

        # Apagar última letra (N)
        if tecla == ord("n") or tecla == ord("N"):
            if palavra_atual:
                palavra_atual = palavra_atual[:-1]

        # Sair (Q)
        if tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# EXECUÇÃO
# ===============================
if __name__ == "__main__":
    main()