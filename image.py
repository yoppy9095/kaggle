from os import replace
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets

from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD

# データ読み込み
train_csv=pd.read_csv("train.csv").replace("male",0).replace("female",1)
x_test=pd.read_csv("test.csv")
y_test=pd.read_csv("gender_submission.csv")


# train,validデータの抽出
y=pd.read_csv("train.csv",usecols=[1])
x=train_csv.drop(['Survived',"Cabin","Age","Ticket"],axis=1)

x=x.dropna(how='all')

#欠損地確認 
# print(x.isnull().sum())

from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid = train_test_split(x,y, train_size=0.8, test_size=0.2)

# # モデル構築
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
#             optimizer=RMSprop(),
            optimizer=SGD(lr=0.1),  
            metrics=['accuracy'])


history = model.fit(x_train, y_train,batch_size=256,epochs=50,verbose=1,validation_data=(x_valid, y_valid))
                        # validation_split=0.2
# model.save("test.h5")

score = model.evaluate(x_test,y_test,  verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])