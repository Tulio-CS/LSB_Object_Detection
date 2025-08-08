import cv2
import mediapipe as mp
import os
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

# Output directory and csv
os.makedirs('data', exist_ok=True)
output_csv = os.path.join('data', 'gesture_data.csv')

header = ['label'] + [f'pose_{i}_{axis}' for i in range(33) for axis in ['x', 'y', 'z', 'visibility']] + \
         [f'left_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']] + \
         [f'right_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])\
                    .flatten() if results.pose_landmarks else np.zeros(33*4)
            left = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])\
                    .flatten() if results.left_hand_landmarks else np.zeros(21*3)
            right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])\
                    .flatten() if results.right_hand_landmarks else np.zeros(21*3)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key != 255:
                label = chr(key)
                writer.writerow(np.concatenate([[label], pose, left, right]))

            cv2.imshow('Gesture Collector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
