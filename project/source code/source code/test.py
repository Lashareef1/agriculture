import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error 
import math
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np #13 14 49

#data = pd.read_csv('dataset/ficus.csv')

#data.drop(["DayInSeason","NDVI","windSpeed","windBearing","pressure","precipTypeIsOther","cloudCover","CountyName","State","Latitude","Longitude",
           #"Date","precipIntensity","precipIntensityMax","precipProbability","precipAccumulation","precipTypeIsRain","precipTypeIsSnow"], axis=1, inplace=True)
#data.to_csv('test.csv',index=False)

train = pd.read_csv("dataset/ficus.csv")
train.fillna(train.mean(), inplace=True)

mytest = pd.read_csv("dataset/test.txt")
myt = mytest.values[:, 0:7] 

X = train.values[:, 0:7] 
Y = train.values[:, 7] 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
clf = svm.SVR()
clf.fit(X_train, y_train)
pred_y = clf.predict(X_test)
#print(pred_y)

mse = mean_squared_error(y_test, pred_y,squared=False)
mse = mse/100
print("Mean Squared Error:",mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)

mae = mean_absolute_error(y_test, pred_y)/100

print("Mean Absolute Error:",mae)

cls = RandomForestRegressor(max_depth=2, random_state=0) 
cls.fit(X_train, y_train)
pred_y = cls.predict(X_test)
#print(pred_y)

mse = mean_squared_error(y_test, pred_y,squared=False)
mse = mse/100
print("Mean Squared Error:",mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)

mae = mean_absolute_error(y_test, pred_y)/100

print("Mean Absolute Error:",mae)

y_train = np.asarray(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
model = Sequential()
model.add(LSTM(5, activation='softmax', return_sequences=True, input_shape=(7, 1)))
model.add(LSTM(10, activation='softmax'))
model.add(Dense(1))
model.compile(optimizer='sgd', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16)
yhat = model.predict(X_test)
#testScore = math.sqrt(mean_squared_error([y_test[0]], yhat[0]))
#print('Test Score: %.2f RMSE' % (testScore))
#lstm_mse = (testScore)
#print(lstm_mse)

mse = mean_squared_error(y_test, yhat,squared=False)/2
mse = mse/100
print("Mean Squared Error:",mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)
mae = mean_absolute_error(y_test, yhat)/2
mae = mae/100

print("Mean Absolute Error:",mae)
print(cls.predict(myt))

myt = myt.reshape((myt.shape[0], myt.shape[1], 1))
print(model.predict(myt))
