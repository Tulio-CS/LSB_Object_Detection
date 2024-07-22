import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the pre-trained model
model = tf.keras.models.load_model('models/holistic/model.h5')
model.load_weights("models/holistic/ModelWeights.weights.h5")

labels = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"L",6:"I",7:"M",8:"V"}
threshold=0.9

encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
# Initialize MediaPipe hands
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

data = "test.csv"

df = pd.read_csv(data)
df = df.iloc[:,1:1600]

df = scaler.transform(df)

prediction = model.predict(df)
predicted_class = np.argmax(prediction, axis=1)[0]
print(predicted_class)