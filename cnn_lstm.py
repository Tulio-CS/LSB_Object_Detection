import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt # Importar matplotlib

# --- Configurações --- 
DATA_DIR = 'libras_data' # Diretório onde os dados coletados estão salvos

# --- Carregamento de Dados ---

sequences, labels = [], []

# Coleta todos os nomes de gestos (subdiretórios em DATA_DIR)
ACTIONS = np.array([name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))])
ACTIONS.sort() # Garante uma ordem consistente das classes

print(f"Classes de gestos detectadas: {ACTIONS}")

for action_idx, action in enumerate(ACTIONS):
    action_path = os.path.join(DATA_DIR, action)
    for file_name in os.listdir(action_path):
        if file_name.endswith(".npy"):
            try:
                res = np.load(os.path.join(action_path, file_name))
                sequences.append(res)
                labels.append(action_idx) # Usa o índice da ação (diretório) como label
            except Exception as e:
                print(f"Erro ao carregar {file_name}: {e}")
                continue

if not sequences:
    print("Nenhum dado encontrado. Certifique-se de que os dados foram coletados e salvos corretamente.")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"Shape total dos dados (X): {X.shape}")
print(f"Shape total dos rótulos (y): {y.shape}")

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% treino, 20% teste

print(f"Shape dos dados de treinamento (X_train): {X_train.shape}")
print(f"Shape dos rótulos de treinamento (y_train): {y_train.shape}")
print(f"Shape dos dados de teste (X_test): {X_test.shape}")
print(f"Shape dos rótulos de teste (y_test): {y_test.shape}")

# --- Construção do Modelo LSTM (sem CNN) ---

model = Sequential()

# Camadas LSTM para processar a sequência de características (landmarks)
# A entrada esperada é (batch_size, timesteps, features_por_timestep).
# X.shape[1] é o número de timesteps (SEQUENCE_LENGTH).
# X.shape[2] é o número de features por timestep (126 landmarks).
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))

# Camada de saída: número de neurônios igual ao número de classes (gestos)
model.add(Dense(ACTIONS.shape[0], activation='softmax'))

# Compila o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Resumo do modelo
model.summary()

# --- Treinamento do Modelo ---
print("\nIniciando o treinamento do modelo...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=50, # Número de épocas de treinamento
    batch_size=32, # Tamanho do batch
    validation_data=(X_test, y_test), # Dados de validação
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)] # Early stopping para evitar overfitting
)

print("Treinamento concluído.")

# --- Avaliação do Modelo ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia do modelo nos dados de teste: {accuracy*100:.2f}%")

# --- Salvando o Modelo ---
model_save_path = 'libras_recognition_model.h5'
model.save(model_save_path)
print(f"Modelo salvo em: {model_save_path}")

# Opcional: Salvar as classes (ACTIONS) para uso posterior na inferência
np.save('libras_actions.npy', ACTIONS)
print("Classes de gestos salvas em: libras_actions.npy")

# --- Plots para visualização do treinamento ---

# Plot da Perda (Loss)
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Curva de Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()

# Plot da Acurácia
plt.figure(figsize=(12, 6))
plt.plot(history.history['categorical_accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_categorical_accuracy'], label='Acurácia de Validação')
plt.title('Curva de Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()