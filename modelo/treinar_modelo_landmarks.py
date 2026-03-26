import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# =========================
# CONFIGURAÇÕES
# =========================
LETRAS = ["A", "B", "C", "D", "E"]
NUM_CLASSES = len(LETRAS)
ENTRADA = 42          # 21 landmarks (x,y)
EPOCHS = 50
BATCH_SIZE = 16

# =========================
# CARREGAR DATASETS
# =========================
def carregar_dataset(letra, label):
    df = pd.read_csv(f"../dataset/dataset_{letra}.csv", header=None)
    X = df.values
    y = np.full((len(X),), label)
    return X, y

Xs, ys = [], []

for idx, letra in enumerate(LETRAS):
    X, y = carregar_dataset(letra, idx)
    Xs.append(X)
    ys.append(y)

X = np.vstack(Xs)
y = np.hstack(ys)

# One-hot
y = to_categorical(y, NUM_CLASSES)

# =========================
# TREINO / TESTE
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# =========================
# MODELO (LANDMARKS)
# =========================
model = Sequential([
    Dense(128, activation="relu", input_shape=(ENTRADA,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TREINAMENTO
# =========================
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =========================
# SALVAR MODELO
# =========================
model.save("../modelo_landmarks.h5")
print("✅ Modelo salvo como modelo_landmarks.h5")