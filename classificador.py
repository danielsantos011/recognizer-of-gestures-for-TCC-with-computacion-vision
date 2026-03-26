import numpy as np
from tensorflow.keras.models import load_model

# ===== CONFIGURAÇÕES =====
MODELO_PATH = "modelo/modelo_landmarks.h5"

LETRAS = [
    "A", "B", "C", "D", "E"
]

CONFIANCA_MIN = 0.80  # ajuste fino: 0.75 a 0.85

# ===== CARREGA O MODELO =====
modelo = load_model(MODELO_PATH)

def classificar(landmarks):
    """
    Recebe landmarks do MediaPipe (21 pontos)
    Retorna letra ou None
    """

    # 🔒 SEGURANÇAS
    if landmarks is None:
        return None

    if len(landmarks) != 21:
        return None

    # 🔹 Converte landmarks em vetor [x1,y1,x2,y2...]
    entrada = []
    for p in landmarks:
        entrada.extend([p.x, p.y])

    # 🔒 Garante formato correto
    if len(entrada) != 42:
        return None

    entrada = np.array(entrada, dtype=np.float32).reshape(1, 42)

    # 🔹 Predição
    try:
        pred = modelo.predict(entrada, verbose=0)[0]
    except Exception as e:
        print("Erro na predição:", e)
        return None

    idx = np.argmax(pred)
    confianca = float(pred[idx])

    # 🔍 DEBUG (opcional – pode comentar depois)
    print("Pred:", pred, "→", LETRAS[idx], f"{confianca:.2f}")

    # ❌ REJEITA GESTO FRACO (palma aberta, confusão, etc.)
    if confianca < CONFIANCA_MIN:
        return None

    return LETRAS[idx]