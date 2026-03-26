import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ===== CARREGAR DADOS =====
A = pd.read_csv("dataset_A.csv")
B = pd.read_csv("dataset_B.csv")
C = pd.read_csv("dataset_C.csv")

# Labels numéricos
A["label"] = 0  # A
B["label"] = 1  # B
C["label"] = 2  # C

dados = pd.concat([A, B, C], ignore_index=True)

# ===== SEPARAR X e Y =====
X = dados.drop("label", axis=1).values  # 42 colunas
y = dados["label"].values

y = to_categorical(y, num_classes=3)

# ===== MODELO =====
modelo = Sequential([
    Dense(128, activation="relu", input_shape=(42,)),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")
])

modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()

# ===== TREINO =====
modelo.fit(
    X, y,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

# ===== SALVAR =====
modelo.save("modelo_landmarks.h5")
print("Modelo salvo com sucesso!")