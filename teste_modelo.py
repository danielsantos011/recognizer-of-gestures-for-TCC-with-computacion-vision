import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo
modelo = load_model("modelo/keras_model.h5")

print("Modelo carregado com sucesso!")

# Criar um dado FAKE só para testar
# Teachable Machine geralmente usa vetores ou imagens normalizadas
entrada_teste = np.random.rand(1, modelo.input_shape[1])

# Fazer previsão
saida = modelo.predict(entrada_teste)

print("Saída do modelo:", saida)