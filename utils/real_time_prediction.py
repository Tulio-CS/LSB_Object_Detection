import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')
model.load_weights("ModelWeights.weights.h5")

labels = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"L",6:"I",7:"M",8:"V"}
threshold=0.9

encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
# Initialize MediaPipe hands
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize MediaPipe drawing


# Open the webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB image to detect hands
        results = holistic.process(rgb_frame)

        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            
            # Right hand
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Left Hand
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Pose Detections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Draw hand landmarks and predict if hand is detected
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                

                #TODO
                #x_min = int(min(x_coords) * frame.shape[1])
                #x_max = int(max(x_coords) * frame.shape[1])
                #y_min = int(min(y_coords) * frame.shape[0])
                #y_max = int(max(y_coords) * frame.shape[0])

                # Draw the bounding box
                #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


                # Extract keypoints
        keypoints = np.concatenate([face,pose,rh]).reshape(1,-1)

        keypoints_scaled = scaler.transform(keypoints)

                # Make a prediction
        prediction = model.predict(keypoints_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]
        cv2.putText(frame,str(predicted_class),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        confidence = prediction[0][predicted_class]

        # Display the resulting frame
        cv2.imshow('Sign Language Predictor', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
