# main.py

import cv2
import threading
import os
from treino_gesto import letra_a
from detector_maos import DetectorMaos
import winsound  # Melhor para WAV no Windows

# Controle para não tocar o áudio repetidamente
audio_tocado = False

def tocar_audio(caminho_audio):
    """Toca o áudio de forma assíncrona"""
    if os.path.exists(caminho_audio):
        winsound.PlaySound(caminho_audio, winsound.SND_ASYNC)
    else:
        print("Arquivo de áudio não encontrado:", caminho_audio)

def main():
    global audio_tocado

    # Caminho absoluto do arquivo de áudio
    raiz = os.path.dirname(os.path.abspath(__file__))  # pasta onde está o main.py
    caminho_audio = os.path.join(raiz, "audios", "letraA.wav")  # arquivo dentro de 'audios'

    if not os.path.exists(caminho_audio):
        print("AVISO: Áudio da letra A não encontrado!")
        return  # Sai do programa se não encontrar o arquivo

    # Captura da webcam
    cap = cv2.VideoCapture(0)
    detector = DetectorMaos()

    while True:
        sucesso, imagem = cap.read()
        if not sucesso:
            break

        imagem = cv2.flip(imagem, 1)  # Espelha a imagem
        imagem = detector.encontrar_maos(imagem)

        letra_detectada = False

        # Verifica se alguma mão foi detectada
        if detector.resultado and detector.resultado.multi_hand_landmarks:
            for mao in detector.resultado.multi_hand_landmarks:
                if letra_a(mao.landmark):
                    letra_detectada = True
                    # Mostra na tela
                    cv2.putText(imagem, "Letra A", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Toca o áudio apenas se a letra foi detectada e ainda não tocou
        if letra_detectada and not audio_tocado:
            threading.Thread(target=tocar_audio, args=(caminho_audio,)).start()
            audio_tocado = True
        # Reseta o controle se a letra sumiu
        elif not letra_detectada:
            audio_tocado = False

        # Mostra a captura
        cv2.imshow("Reconhecimento de Libras", imagem)

        # Sai ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()