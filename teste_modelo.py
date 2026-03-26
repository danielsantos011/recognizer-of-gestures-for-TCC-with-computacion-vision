import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

LETRAS = ["A", "B", "C", "D", "M", "N"]

X = []
y = []

for i, letra in enumerate(LETRAS):
    dados = pd.read_csv(f"dataset_{letra}.csv", header=None).values
    X.append(dados)
    y.append(np.full(len(dados), i))

X = np.vstack(X)
y = np.hstack(y)

y = to_categorical(y, num_classes=len(LETRAS))

# ===== MODELO =====
modelo = Sequential([
    Dense(128, activation="relu", input_shape=(42,)),
    Dense(64, activation="relu"),
    Dense(len(LETRAS), activation="softmax")
])

modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.fit(X, y, epochs=40, batch_size=16)

modelo.save("modelo_landmarks.h5")
print("Modelo salvo com sucesso!")