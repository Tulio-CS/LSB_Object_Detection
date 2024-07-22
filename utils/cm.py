import seaborn as sn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

#Variaveis
data_path = "holistic.csv"     #Caminho do dataset
seed = 13


model = tf.keras.models.load_model('model.h5')
model.load_weights("ModelWeights.weights.h5")


#Criando o dataset

data_frame = pd.read_csv(data_path,sep=",",decimal=".")
pred_ds = data_frame.sample(frac=0.5,random_state=seed)

encoder = joblib.load("encoder.pkl")
pred_ds["target"] = encoder.transform(pred_ds["y"])
scaler = joblib.load("scaler.pkl")
pred_ds.iloc[:,1:259] = pd.DataFrame(scaler.transform(pred_ds.iloc[:,1:259]))

y_pred = model.predict(pred_ds.iloc[:,1:259])

truth = pred_ds["target"]

#Criando a matriz de confusão
predictions = np.argmax(y_pred,axis=1)

cm = tf.math.confusion_matrix(labels=truth, predictions=predictions)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Truth')
plt.xlabel('Predicted')

plt.show()