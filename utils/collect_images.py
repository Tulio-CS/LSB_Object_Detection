import cv2
import mediapipe as mp
import os
import csv
from collections import defaultdict

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# Dictionary to store image counters for each key
image_counters = defaultdict(int)

# Create a base directory to save images
base_dir = 'C:/Users/tulio/OneDrive/Documentos/GitHub/lsb_images'
os.makedirs(base_dir, exist_ok=True)

# CSV file setup
csv_file = 'dataset.csv'
csv_header = ['y'] + [f'keypoint_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

# Open the CSV file for writing
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

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

        # Draw hand landmarks and save image if hand is detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key != 255:  # If a valid key is pressed
                    key_str = chr(key)
                    key_dir = os.path.join(base_dir, key_str)

                    #SAVE IMAGE
                    
                    os.makedirs(key_dir, exist_ok=True)
                    image_counters[key_str] += 1
                    image_path = os.path.join(key_dir, f"{image_counters[key_str]:04d}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"Saved {image_path}")

                    # Save keypoints to CSV
                    keypoints = [key_str]
                    for landmark in hand_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.z])
                    writer.writerow(keypoints)

        # Display the resulting frame
        cv2.imshow('Hand Detector', frame)

        # Check for exit condition after showing the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
