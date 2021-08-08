# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:07:53 2021

@author: Andres
"""

# Predicción de Valor de acciones de Amazon


## 1 - Importar datos
##Importo el dataset y grafico los valores.

# Clonamos el repositorio que contiene las dataset de ventas mensuales de la aerolínea.

! git clone https://github.com/AndresTill/LSTM.git

import pandas
import numpy
import matplotlib.pyplot as plt
from datetime import datetime

raw_data = pandas.read_csv('LSTM/Valor accion Amazon 2010-2020.csv', usecols=[1])
plt.plot(raw_data)
plt.show()

#Observemos algunos datos
raw_data.head(10)

## 2 - Preprocesamiento de datos
##Ahora normalizaremos los datos con el [MinMaxScaler de sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

### SOLUCIÓN
from sklearn.preprocessing import MinMaxScaler

dataset = raw_data.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)



### SOLUCIÓN
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]



### SOLUCIÓN
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-2):
        dataX.append([dataset[i, 0]])
        dataY.append(dataset[i + 1, 0])
    return numpy.array(dataX), numpy.array(dataY)

##Defino un dataset mirando únicamente el valor de las acciones del dia previo para calcular el valor del dia siguiente.

trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)


trainX.shape

trainY.shape

# Agrego una dimensión para que los datos tengan la forma necesaria para más adelante
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

## 3 - Red neuronal
##Crear una red neuronal con una capa [LSTM](https://keras.io/layers/recurrent/#lstm) y una capa densa. Utilizar la función de pérdida mean_squared_error.


### SOLUCIÓN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))

model.summary()

##A continuación debemos compilar y entrenar el modelo.

### SOLUCIÓN
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

## 4 - Resultados
##Es momento de analizar los resultados obtenidos. Utilicemos el modelo para predecir las ventas sobre los datos de train y test.

### SOLUCIÓN
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

##Debemos invertir la normalización hecha sobre los datos para poder calcular el error sobre el valor real de los datos.

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

##Ahora tomemos una medida objetiva del error como el error cuadrático medio.

from sklearn.metrics import mean_squared_error
import math

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test: %.2f RMSE' % (testScore))

# Preparo las predicciones sobre los datos de training para graficar
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict

# Preparo las predicciones sobre los datos de test para graficar
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(2)+1:len(dataset)-1, :] = testPredict

# Grafico
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

