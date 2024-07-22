import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the pre-trained model
model = tf.keras.models.load_model('models/One_Hand.h5')
model.load_weights("models\One_Hand.weights.h5")

labels = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"I",6:"L",7:"M",8:"N",9:"O",10:"U",11:"V",12:"W"}
threshold=0.9

encoder = joblib.load("encoders/One_Hand_encoder.pkl")
scaler = joblib.load("encoders/One_Hand_scaler.pkl")
# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image to detect hands
    result = hands.process(rgb_frame)

    # Draw hand landmarks and predict if hand is detected
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:
            # Calculate the bounding box
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            z_coords = [landmark.z for landmark in hand_landmarks.landmark]
            
            x_min = int(min(x_coords) * frame.shape[1])
            x_max = int(max(x_coords) * frame.shape[1])
            y_min = int(min(y_coords) * frame.shape[0])
            y_max = int(max(y_coords) * frame.shape[0])

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


            # Extract keypoints
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            
            keypoints = np.array(keypoints).reshape(1, -1)  # Reshape for the model
            keypoints_scaled = scaler.transform(keypoints)

            # Make a prediction
            prediction = model.predict(keypoints_scaled)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class]
            
            if confidence >= threshold:

                # Display the predicted label on the frame
                label_position = (x_min + 10, y_min + 30)  # Position inside the bounding box
                cv2.putText(frame, labels[predicted_class], label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Sign Language Predictor', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
