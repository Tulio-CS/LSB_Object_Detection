import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# --- Configurações ---
MODEL_PATH = 'libras_recognition_model.h5'
ACTIONS_PATH = 'libras_actions.npy'
SEQUENCE_LENGTH = 30 # Deve ser o mesmo usado na coleta de dados e treinamento

# --- Carregar o modelo e as classes ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    ACTIONS = np.load(ACTIONS_PATH)
    print(f"Modelo e classes carregados com sucesso. Classes: {ACTIONS}")
except Exception as e:
    print(f"Erro ao carregar o modelo ou as classes: {e}")
    print("Certifique-se de que o modelo foi treinado e salvo corretamente (execute train_model.py primeiro).")
    exit()

# --- Inicializar MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# --- Inicializar Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

# --- Variáveis para inferência em tempo real ---
sequence = []
predictions = []
threshold = 0.7 # Limiar de confiança para exibir a previsão

print("Iniciando o teste do modelo. Pressione 'q' para sair.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Espelhar o frame para visualização consistente com a coleta de dados
    frame_flipped = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    current_frame_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenhar landmarks no frame espelhado
            mp_drawing.draw_landmarks(
                frame_flipped,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            for landmark in hand_landmarks.landmark:
                current_frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

    # Preencher com zeros se não houver landmarks detectados ou se houver menos de 2 mãos
    # Garante que o vetor de landmarks tenha sempre 126 valores (21 landmarks * 3 coords * 2 mãos)
    if len(current_frame_landmarks) < 126:
        current_frame_landmarks.extend([0.0] * (126 - len(current_frame_landmarks)))
    
    # Adicionar os landmarks do frame atual à sequência
    sequence.append(current_frame_landmarks[:126])
    sequence = sequence[-SEQUENCE_LENGTH:] # Manter apenas os últimos SEQUENCE_LENGTH frames

    # Se a sequência tiver o tamanho necessário, faça a previsão
    if len(sequence) == SEQUENCE_LENGTH:
        # Reshape para (1, SEQUENCE_LENGTH, 126) para o modelo
        input_data = np.expand_dims(sequence, axis=0)
        
        # Fazer a previsão
        res = model.predict(input_data)[0]
        predicted_action_index = np.argmax(res)
        confidence = res[predicted_action_index]

        # Exibir a previsão se a confiança for alta o suficiente
        if confidence > threshold:
            predicted_action = ACTIONS[predicted_action_index]
            text_to_display = f'{predicted_action} ({confidence*100:.2f}%)'
        else:
            text_to_display = '...' # Ou 'Nenhum sinal detectado'
        
        # Adicionar a previsão à lista de previsões para suavização (opcional)
        predictions.append(predicted_action_index)
        predictions = predictions[-10:] # Manter as últimas 10 previsões

        # Exibir o texto na tela
        cv2.putText(frame_flipped, text_to_display, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Teste do Modelo Libras', frame_flipped)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Teste do modelo encerrado.")