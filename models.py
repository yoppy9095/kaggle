from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def random_forest_model(params):
    # # 学習に関するパラメータ設定
    # params = {
    #     "n_estimators" : [2, 5, 10, 15, 20, 30, 50, 75, 100, 200, 500, 1000],
    #     "criterion" : ["gini"],
    #     "min_samples_split" : [2, 3, 5, 10, 15, 20, 30],
    #     "max_depth" : [2, 3, 5, 10, 15, 20, 30],
    #     "random_state" : [1],
    #     "verbose" : [False],
    # }

    # モデル構築
    model = GridSearchCV(RandomForestClassifier(), params, cv = 3)

    return model



def nn_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=4))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model