from keras import models, layers, optimizers
from keras.metrics import MeanSquaredError, MeanAbsoluteError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Read the test and train files
df_test = pd.read_csv('df_test_predict.csv')
df_train = pd.read_csv('df_train .csv')

# print(df_train.describe())
# print(df_test.describe())

# joins both files into one to pre-processing the nominal categorical values
# The last 2500 variables are the train set, so we can split them apart later.
X = pd.concat([df_train, df_test])
Y = X.iloc[:, 29]
X = X.iloc[:, :29]

# Cast categorical values into str
X['appType'] = X['appType'].astype(str)
X['creatSize'] = X['creatSize'].astype(str)
X['creatType'] = X['creatType'].astype(str)
X['deviceOs'] = X['deviceOs'].astype(str)
X['domain'] = X['domain'].astype(str)
X['tsDow'] = X['tsDow'].astype(str)

# See the NaN values in domain
'''
    x = X['domain'].values
    for i in range(len(x)):
      if x[i] == 'nan':
        print('exists in pos ', i)
    '''

# Handling the NaN values
imputer = SimpleImputer(missing_values='', strategy='most_frequent')
x = X['domain'].values
X['domain'] = imputer.fit_transform(x.reshape(-1, 1))

# Verify if NaN values passed to str
'''
    x = X['domain'].values
    for i in range(len(x)):
      if x[i] == 'nan':
        print('exists in pos ', i)
    '''

# To transform textual predictors into numeric variables
labelencoder = LabelEncoder()
X['appType'] = labelencoder.fit_transform(X['appType'])
X['creatSize'] = labelencoder.fit_transform(X['creatSize'])
X['creatType'] = labelencoder.fit_transform(X['creatType'])
X['deviceOs'] = labelencoder.fit_transform(X['deviceOs'])
X['domain'] = labelencoder.fit_transform(X['domain'])
X['tsDow'] = labelencoder.fit_transform(X['tsDow'])

# Since they are non-measurable data, dummy variables are necessary, for a category with a greater number not to have more weight than one with a lower one.
X = pd.get_dummies(
    X, columns=['appType', 'creatSize', 'creatType', 'deviceOs', 'domain', 'tsDow'])

# Parameters scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reparting the test and training set as it was initially
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, train_size=10000, random_state=None, shuffle=False)

print('Reading and pre-processing done!')

# Ceates the model
model= models.Sequential()

# Adding the layers (6 hidden layers)
model.add(layers.Dense(512, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(512, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(256, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(256, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(128, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(128, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))

model.summary()

#Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

#Training the model
history = model.fit(x_train, y_train, epochs=30, validation_split=0.3)

print("Model trained!")

# list all data in history
#print(history.history.keys())

#see the training process graphically
loss = history.history['loss']
val_mse = history.history['val_mean_squared_error']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss with MSE')
plt.plot(epochs, val_mse, 'b', label='Training validation with MSE')
plt.title('Training loss and validation with MSE')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.legend()
plt.show()

print("Training graphic shown!")

#Make the prediction for the test set
prediction = model.predict(x_test)
print(prediction)

#Save the prediction results in a new CSV file
prediction = pd.DataFrame(prediction)
prediction.to_csv('pred.csv')

print("Prediction done and saved!")