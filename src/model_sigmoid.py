import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/Churn_Modelling.csv')
#print(df.head())

df.drop(columns=['CustomerId','RowNumber','Surname'],inplace=True)

# OHE
df = pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)

# SCALED VALUES
X = df.drop(columns=['Exited'])
y = df['Exited']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# PREPROCESSING
scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# APPLY ANN

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Build model(2 types of model can be made using keras 1.> sequential 2.> non-sequential)
# Building sequential model

model = Sequential()

# create model
# adding layer :- 1 - i/p, 1 - o/p, 1 - hidden layer,3 - nodes(perceptron)
model.add(Dense(3,activation = 'sigmoid',input_dim = 11)) #hidden layer # i/p_dimension -> how many i/p it will be getting no of columns
model.add(Dense(1,activation = 'sigmoid')) # o/p layer 

model.summary()

# compile model
model.compile(loss = 'binary_crossentropy',optimizer='Adam')

# model train
model.fit(X_trained_scaled,y_train,epochs = 10)
model.layers[0].get_weights() # weights and biases of model
y_log = model.predict(X_test_scaled)
y_pred = np.where(y_log>0.5,1,0)

# accuracy

from sklearn.metrics import accuracy_score
print()
print(accuracy_score(y_test,y_pred))
