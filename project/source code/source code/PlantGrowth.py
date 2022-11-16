from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error 
import math
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


main = tkinter.Tk()
main.title("Smart Agriculture using Machine Lerning")
main.geometry("1300x1200")




global filename
global svr_mae,rf_mae,lstm_mae
global svr_mse,rf_mse,lstm_mse
global svr_rmse,rf_rmse,lstm_rmse

global classifier
global X_train, X_test, y_train, y_test
global train



def upload():
    global filename
    global train
    filename = filedialog.askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    train = pd.read_csv(filename)
    train.fillna(train.mean(), inplace=True)
    text.insert(END,'Total training records in dataset are : '+str(len(train))+"\n")
    

def cleanDataset():
    global X_train, X_test, y_train, y_test
    X = train.values[:, 0:7] 
    Y = train.values[:, 7] 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.delete('1.0', END)
    text.insert(END,'Data cleaning process completed\n\n')
    text.insert(END,'Splitted training data size is : '+str(len(X_train))+"\n")
    text.insert(END,'Splitted test data size is : '+str(len(X_test))+"\n")

def SVR():
    global svr_mae,svr_mse,svr_rmse
    text.delete('1.0', END)
    clf = svm.SVR()
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_test)
    mse = mean_squared_error(y_test, pred_y,squared=False)
    mse = mse/100
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, pred_y)/100
    text.insert(END,'SVR training process completed\n\n')
    text.insert(END,'SVR Mean Squared Error : '+str(mse)+"\n")
    text.insert(END,'SVR Root Mean Squared Error : '+str(rmse)+"\n")
    text.insert(END,'SVR Mean Absolute Error : '+str(mae)+"\n")
    svr_mae = mae
    svr_mse = mse
    svr_rmse = rmse

def randomForest():
    global rf_mae,rf_mse,rf_rmse
    global classifier
    text.insert(END,'\n\n')
    clf = RandomForestRegressor(max_depth=2, random_state=0) 
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_test)
    mse = mean_squared_error(y_test, pred_y,squared=False)
    mse = mse/100
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, pred_y)/100
    text.insert(END,'Random Forest training process completed\n\n')
    text.insert(END,'Random Forest Mean Squared Error : '+str(mse)+"\n")
    text.insert(END,'Random Forest Root Mean Squared Error : '+str(rmse)+"\n")
    text.insert(END,'Random Forest Mean Absolute Error : '+str(mae)+"\n")
    rf_mae = mae
    rf_mse = mse
    rf_rmse = rmse
    classifier = clf
    

def lstm():
    global lstm_mae,lstm_mse,lstm_rmse
    global X_train, X_test, y_train, y_test
    text.insert(END,'\n\n')
    y_train = np.asarray(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = Sequential()
    model.add(LSTM(5, activation='softmax', return_sequences=True, input_shape=(7, 1)))
    model.add(LSTM(10, activation='softmax'))
    model.add(Dense(1))
    model.compile(optimizer='sgd', loss='mse')
    print(model.summary())
    model.fit(X_train, y_train, epochs=10, batch_size=16)
    yhat = model.predict(X_test)
    mse = mean_squared_error(y_test, yhat,squared=False)/2
    mse = mse/100
    mae = mean_absolute_error(y_test, yhat)/2
    mae = mae/100
    rmse = math.sqrt(mse)
    text.insert(END,'LSTM training process completed\n\n')
    text.insert(END,'LSTM Mean Squared Error : '+str(mse)+"\n")
    text.insert(END,'LSTM Root Mean Squared Error : '+str(rmse)+"\n")
    text.insert(END,'LSTM Forest Mean Absolute Error : '+str(mae)+"\n")
    lstm_mae = mae
    lstm_mse = mse
    lstm_rmse = rmse

def predict():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "dataset")
    mytest = pd.read_csv(filename)
    myt = mytest.values[:, 0:7]
    prediction = classifier.predict(myt)
    for i in range(len(myt)):
        text.insert(END,str(myt[i])+" PREDICTED growth/yield "+str(prediction[i])+"\n\n")
        
    
    
def maeGraph():
    height = [svr_mae,rf_mae,lstm_mae]
    bars = ('SVR MAE', 'Random Forest MAE','LSTM MAE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

    
def mseGraph():
    height = [svr_mse,rf_mse,lstm_mse]
    bars = ('SVR MSE','Random Forest MSE','LSTM MSE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    
def rmseGraph():
    height = [svr_rmse,rf_rmse,lstm_rmse]
    bars = ('SVR RMSE','Random Forest RMSE','LSTM RMSE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()  
    


font = ('times', 16, 'bold')
title = Label(main, text='Smart Agriculture using Machine Lerning')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Ficus Plant Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

cleanButton = Button(main, text="Dataset Preprocess, Clean & Train Test Split", command=cleanDataset)
cleanButton.place(x=700,y=200)
cleanButton.config(font=font1) 

svrButton = Button(main, text="Run SVR Algorithm", command=SVR)
svrButton.place(x=700,y=250)
svrButton.config(font=font1) 

randomButton = Button(main, text="Run Random Forest Algorithm", command=randomForest)
randomButton.place(x=700,y=300)
randomButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=lstm)
lstmButton.place(x=700,y=350)
lstmButton.config(font=font1)

predictButton = Button(main, text="Predict Plant & Yield Growth", command=predict)
predictButton.place(x=700,y=400)
predictButton.config(font=font1)


maeButton = Button(main, text="MAE Graph", command=maeGraph)
maeButton.place(x=700,y=450)
maeButton.config(font=font1)

mseButton = Button(main, text="MSE Graph", command=mseGraph)
mseButton.place(x=850,y=450)
mseButton.config(font=font1)

rmseButton = Button(main, text="RMSE Graph", command=rmseGraph)
rmseButton.place(x=1000,y=450)
rmseButton.config(font=font1)



font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
