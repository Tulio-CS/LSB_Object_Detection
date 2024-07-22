# Tulio Castro Silva


#importar as bibliotecas
import tensorflow as tf
from keras import layers, models,callbacks
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from tkinter.filedialog import askopenfile
import seaborn as sn
from random import randint
import joblib


path = "one_hand.csv"
seed = 13
epocas = 500
otimizador = "Adam"

#Crir o dataframe
data_frame = pd.read_csv(path,sep=",",decimal=".")





labels = data_frame["y"].unique()                             #Colhendo os diferentes labels do data frame
encoder = LabelEncoder()                                      #Criando o codificador
encoder.fit(labels)                                           #Ajustando o codificador
data_frame["target"] = encoder.transform(data_frame["y"])      #Criando uma nova coluna no data frame


joblib.dump(encoder,"encoder.pkl")



#Ajustando o data frame
scaler = StandardScaler()           #Criando o scaler
scaler.fit(data_frame.iloc[:,1:1600])
data_frame.iloc[:,1:1600] = pd.DataFrame(scaler.fit_transform(data_frame.iloc[:,1:1600]))           #Normalizando os valores

joblib.dump(scaler,"scaler.pkl")


#Criando os valores de treino, teste e validacao 



train = data_frame.sample(frac=0.8,random_state=seed)         #Separando 80% do data frame para o treino
data_frame = data_frame.drop(train.index)                     #Removendo os valores de treino do data frame
valid = data_frame.sample(frac=0.5,random_state=seed)         #Separando 50% dos valores para a validacao
data_frame = data_frame.drop(valid.index)                     #Removendo os valores de validacao do data frame
test = data_frame.sample(frac=1,random_state=seed)            #Separando o resto dos valores para o teste



X_train = train.iloc[:,1:1600]                 
y_train = train.iloc[:,1600]
X_valid = valid.iloc[:,1:1600]
y_valid = valid.iloc[:,1600]
X_test = test.iloc[:,1:1600]
y_test = test.iloc[:,1600]

#Criando o modelo

tf.random.set_seed(seed)

model = models.Sequential()

model.add(layers.Dense(1599,activation="relu",input_dim = 1599))

model.add(layers.Dense(26,activation="relu"))
model.add(layers.Dense(39,activation="relu"))
model.add(layers.Dense(26,activation="relu"))

model.add(layers.Dense(9,activation="softmax"))

#Otimizador
model.compile(loss='sparse_categorical_crossentropy', optimizer=otimizador, metrics=['accuracy'])

#model.summary()


#Criando o checkpoint, para salvar os melhores pesos
callback = callbacks.ModelCheckpoint("best.keras",save_best_only=True)

reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=3, min_lr=1e-6)

#Criando uma condicao para que a rede pare de treinar se nao houver melhoras, ajuda a evitar overfitting
early_stopping_callback = callbacks.EarlyStopping(monitor="accuracy",patience=10,restore_best_weights=True)         


#Treinamento
history = model.fit(X_train, y_train, 
                    epochs=epocas, 
                    batch_size=64, 
                    validation_data=(X_valid, y_valid), 
                    shuffle=True, 
                    callbacks=[early_stopping_callback, callback, reduce_lr_callback],
                    verbose=1)
#Treinando o modelo


#Carregando os melhores pesos
model.load_weights("best.keras")

#Salvando o modelo
model.save("model.h5")

#Salvando os pesos
model.save_weights("ModelWeights.weights.h5")

#Plotando o grafico de acuracia
plt.plot(history.history['accuracy'],color='red',label='training accuracy')
plt.plot(history.history['val_accuracy'],color='blue',label='validation accuracy')
plt.legend()
plt.show()

#Plotando o grafico de loss
plt.plot(history.history['loss'],color='red',label='training loss')
plt.plot(history.history['val_loss'],color='blue',label='validation loss')
plt.legend()
plt.show()


