from os import replace
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import tensorflow as tf
# from tensorflow import keras
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets

# from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import SGD

# データ読み込み
train_csv=pd.read_csv("train.csv").replace("male",0).replace("female",1).replace("S",0).replace("Q",1).replace("C",2)
test_csv=pd.read_csv("test.csv").replace("male",0).replace("female",1).replace("S",0).replace("Q",1).replace("C",2)
y_test_data=pd.read_csv("gender_submission.csv")


# train,validデータの抽出
y=pd.read_csv("train.csv",usecols=[1])
x=train_csv.drop(["PassengerId","Name",'Survived',"Cabin","Age","Ticket"],axis=1)
x_test=test_csv.drop(["PassengerId","Name","Cabin","Age","Ticket"],axis=1)
y_test=y_test_data.drop(["PassengerId"],axis=1)


# x_test=x_test.dropna(how="any")
# x=x.dropna(how='any')

x_test=x_test.fillna(x_test.mean())
x=x.fillna(x.mean())

#欠損地確認 
# print(x.isnull().sum())
# print(x_test.isnull().sum())

from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid = train_test_split(x,y, train_size=0.8, test_size=0.2)


y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)
y_valid = tf.keras.utils.to_categorical(y_valid, 2)

# モデル構築
model = Sequential()

model.add(Dense(256, input_dim = 6,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
            # optimizer=RMSprop(),
            optimizer=SGD(lr=0.01),  
            metrics=['accuracy'])


history = model.fit(x_train, y_train,batch_size=256,epochs=1000,verbose=1,validation_data=(x_valid, y_valid))
                        
model.save("test.h5")

score = model.evaluate(x_valid, y_valid,  verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

new_model = tf.keras.models.load_model("test.h5")
new_model.summary()


predictions = new_model.predict(x_test)
# print(predictions)
# print( y_test)

count = 0 
i = 0
y_test_data=pd.read_csv("gender_submission.csv")
y_test=y_test_data.drop(["PassengerId"],axis=1)


# for i in range(418):
#     print(i,np.argmax(predictions[i]))

for item in y_test["Survived"]:
    predict_data = np.argmax(predictions[i])

    print(predict_data," t ",item)


    if predict_data == item:

        count += 1
    i+=1

# テストデータ出力
print(count/418)
