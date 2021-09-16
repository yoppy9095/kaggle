import numpy as np
import pandas as pd

train = pd.read_csv("titanic/train.csv", index_col=0).replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
test = pd.read_csv("titanic/test.csv", index_col=0).replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
#csvデータの読み込み


#print(train.head()) #先頭5データを出力
#print(train.describe()) 基本統計量
def loss_table(df):
    null_val = df.isnull().sum()
    percent = 100 * null_val/len(df)
    loss_table = pd.concat([null_val, percent], axis=1)
    loss_table_ren_columns = loss_table.rename(
    columns = {0 : "欠損数", 1 : "%"})
    return loss_table_ren_columns

#print(loss_table(train))
#print(loss_table(test))

#欠損データの確認

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
#欠損データの補完

#print(loss_table(train))
#欠損データ保管の確認



print(train.head(10))