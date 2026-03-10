# treino_gestos.py

from utils import dedo_dobrado

def letra_a(landmarks):
    """
    Reconhecimento da letra A em Libras
    Mão fechada com polegar para foraa
    """

    # Dedos dobrados
    indicador = dedo_dobrado(landmarks[5], landmarks[8])
    medio      = dedo_dobrado(landmarks[9], landmarks[12])
    anelar     = dedo_dobrado(landmarks[13], landmarks[16])
    minimo     = dedo_dobrado(landmarks[17], landmarks[20])

    # Polegar para o lado
    polegar_lado = abs(landmarks[4].x - landmarks[2].x) > 0.04

    if indicador and medio and anelar and minimo and polegar_lado:
        return True

    return False