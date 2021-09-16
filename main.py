from options import opt_args
from image import titanic_train, titanic_test
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def main():
    #option
    opts = opt_args()

    #データセット
    x_train, y_train = titanic_train(opts.train_path)

    #学習用と検証用に分割
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.5, random_state = 1)

    # 学習に関するパラメータ設定
    params = {
        "n_estimators" : [2, 5, 10, 15, 20, 30, 50, 75, 100, 200, 500, 1000],
        "criterion" : ["gini"],
        "min_samples_split" : [2, 3, 5, 10, 15, 20, 30],
        "max_depth" : [2, 3, 5, 10, 15, 20, 30],
        "random_state" : [1],
        "verbose" : [False],
    }

    # モデル構築
    model = GridSearchCV(RandomForestClassifier(), params, cv = 3)
    model = model.fit(x_train, y_train)
    model = model.best_estimator_

    # 検証用セットを用いて評価
    print(model.score(x_val, y_val))








if __name__ == '__main__':
    main()