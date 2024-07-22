import cv2
import mediapipe as mp
import os
import csv
from collections import defaultdict
import numpy as np
import pandas as pd

# Initialize MediaPipe drawing
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# Open the webcam
cap = cv2.VideoCapture(0)

base_dir = 'C:/Users/tulio/OneDrive/Documentos/GitHub/lsb_images_2'
os.makedirs(base_dir, exist_ok=True)

csv_file = 'dataset.csv'
counter = 0
last_char = ""
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(image)

            # Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            
            # Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                            
            

            
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

                # Check for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key != 255:  # If a valid key is pressed
                key_str = chr(key)
                key_dir = os.path.join(base_dir, key_str)

                if key_str != last_char:
                    last_char = key_str
                else:
                    counter += 1
                    print(f"{key_str}  {counter}")
                writer.writerow(np.concatenate([[key_str],face,pose,lh,rh]))

                # Display the resulting frame
            cv2.imshow('Hand Detector', frame)

                # Check for exit condition after showing the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


