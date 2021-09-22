import numpy as np
import pandas as pd


def titanic_train(dir):
    # train_path = "D:/kaggle/titanic/train.csv"
    train = pd.read_csv(dir, index_col=0).replace("male",0).replace("female",1).replace("S",1).replace("C",2).replace("Q",3)

    drop_columns = ["Name","Cabin","Fare","Ticket"]
    train_modify = train.drop(drop_columns, axis=1)
    train_modify = train_modify.dropna(subset=['Age'])
    train_modify = train_modify.dropna(subset=['Embarked'])
    y_train = train_modify["Survived"]
    x_train = train_modify.drop(labels = ["Survived"], axis = 1)

    return x_train, y_train


def titanic_test(t_dir, g_dir):
    # test_path = "D:/kaggle/titanic/test.csv"
    x_test = pd.read_csv(t_dir, index_col=0).replace("male",0).replace("female",1).replace("S",1).replace("C",2).replace("Q",3)
    y_test = pd.read_csv(g_dir, index_col=0)

    drop_columns = ["Name","Cabin","Fare","Ticket"]
    x_test = x_test.drop(drop_columns, axis=1)

    return x_test, y_test


#デバッグ用
# print(titanic_train("D:/kaggle/titanic/train.csv").shape)
# print(titanic_test("D:/kaggle/titanic/test.csv").shape)