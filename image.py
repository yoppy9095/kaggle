import numpy as np
import pandas as pd


def titanic_train(dir):
    # train_path = "D:/kaggle/titanic/train.csv"
    train = pd.read_csv(dir, index_col=0).replace("male",0).replace("female",1)

    drop_columns = ["Name","Age","Cabin","Fare","Embarked","Ticket"]
    train_modify = train.drop(drop_columns, axis=1)
    
    y_train = train_modify["Survived"]
    x_train = train_modify.drop(labels = ["Survived"], axis = 1)

    return x_train, y_train


def titanic_test(dir):
    # test_path = "D:/kaggle/titanic/test.csv"
    test = pd.read_csv(dir, index_col=0)


    drop_columns = ["Age","Cabin","Fare","Embarked"]
    test_modify = test.drop(drop_columns, axis=1)

    return test_modify


#デバッグ用
# print(titanic_train("D:/kaggle/titanic/train.csv").shape)
# print(titanic_test("D:/kaggle/titanic/test.csv").shape)