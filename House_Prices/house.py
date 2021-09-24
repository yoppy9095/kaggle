import numpy as np
import pandas as pd

# csvデータの読み込み
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# # 欠損データの確認
# loss_train = train.isnull().sum()[train.isnull().sum()>0]
# print(loss_train)
# loss_test = test.isnull().sum()[test.isnull().sum()>0]
# print(loss_test)

# 欠損値の補完
def loss_com(df):
    num2str_list = ['MSSubClass','YrSold','MoSold']
    for column in num2str_list:
        df[column] = df[column].astype(str)

    for column in df.columns:
    # dtypeがobjectの場合、文字列の変数
        if df[column].dtype=='O':
            df[column] = df[column].fillna('None')
    # dtypeがint , floatの場合、数字の変数
        else:
            df[column] = df[column].fillna(0)
    
    return df

train_com = loss_com(train)
train_com = loss_com(test)

# 特徴量エンジニアリング
def add_columns(df):
    
    #総面積
    df["total_SF"] = df["1stFlrSF"] + df["2stFlrSF"] + df["TotalBsmtSF"]

    df[""]