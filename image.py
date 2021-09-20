import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Lambda, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import plot_model, to_categorical
import matplotlib.pyplot as plt
from matplotlib import cm

#csvデータの読み込み
train = pd.read_csv("titanic/train.csv", index_col=0).replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
test = pd.read_csv("titanic/test.csv", index_col=0).replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
answer = pd.read_csv("titanic/gender_submission.csv", index_col=0)


#print(train.head()) #先頭5データを出力
#print(train.describe()) 基本統計量
def loss_table(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val/len(df)
    loss_table = pd.concat([null_val, percent], axis=1)
    loss_table_ren_columns = loss_table.rename(
    columns = {0 : "欠損数", 1 : "%"})
    return loss_table_ren_columns


#欠損データの確認
#print(loss_table(train))
#print(loss_table(test))


#欠損データの補完
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(0)

#欠損データ補完の確認
#print(loss_table(train))


#print(train.head(10))

#テストデータの補完
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

#print(loss_table(test))

#必要データの抽出
drop_columns = ["Name","Cabin","Fare","Ticket"]
train_modify = train.drop(drop_columns, axis=1)
y_train = train_modify["Survived"]
x_train = train_modify.drop(labels = ["Survived"], axis = 1)

test_modify = test.drop(drop_columns, axis=1)


#モデル構築
# model = Sequential()

# model.add(Dense(256, input_dim = 6))
# model.add(Activation('relu'))

# model.add(Dense(256))
# model.add(Activation('relu'))

# model.add(Dense(256))
# model.add(Activation('relu'))

# model.add(Dense(256))
# model.add(Activation('relu'))


# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# adam = optimizers.Adam(lr=1e-4)
# model.compile(optimizer=adam, 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

#実行
history = model.fit(x_train, y_train, batch_size=128, epochs=500, verbose=1, validation_split=0.1)

# model.save("model.h5")

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.grid()
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

new_model = tf.keras.models.load_model("model.h5")
new_model.summary()

predictions = new_model.predict(test_modify)



count = 0 
i = 0
for item in answer["Survived"]:
    predict_data = np.argmax(predictions[i])
    if predict_data == item:
        count += 1
    i+=1
#テストデータ出力
print(count/len(answer))
