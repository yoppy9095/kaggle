import numpy as np

#unko

#aaa


import pandas as pd

#kkkkkk


import matplotlib.pyplot as plt

def titanic_train(dir):
    # train_path = "D:/kaggle/titanic/train.csv"
    train = pd.read_csv(dir, index_col=0)


    drop_columns = ["Age","Cabin","Fare","Embarked"]
    train_modify = train.drop(drop_columns, axis=1)

    return train_modify


def titanic_test(dir):
    # test_path = "D:/kaggle/titanic/test.csv"
    test = pd.read_csv(dir, index_col=0)


    drop_columns = ["Age","Cabin","Fare","Embarked"]
    test_modify = test.drop(drop_columns, axis=1)

    return test_modify


#デバッグ用
# print(titanic_train("D:/kaggle/titanic/train.csv").shape)
# print(titanic_test("D:/kaggle/titanic/test.csv").shape)