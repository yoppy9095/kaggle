from options import opt_args
from image import titanic_train, titanic_test
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import optimizers

import pandas as pd
import matplotlib.pyplot as plt
from models import random_forest_model, nn_model




def main():
    #option
    opts = opt_args()

    #データセット
    x_train, y_train = titanic_train(opts.train_path)

    #学習用と検証用に分割
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.5, random_state = 1)

    # 学習に関するパラメータ設定
    params = {
        "n_estimators" : [2, 5, 10, 15, 20, 30, 50, 75, 100, 200, 500, 1000],
        "criterion" : ["gini"],
        "min_samples_split" : [2, 3, 5, 10, 15, 20, 30],
        "max_depth" : [2, 3, 5, 10, 15, 20, 30],
        "random_state" : [1],
        "verbose" : [False],
    }


    if opts.model_name == 'random_forest':
        
        model = random_forest_model(params)
        model = model.fit(x_train, y_train)

        # スコアの一覧を取得
        model_result = pd.DataFrame.from_dict(model.cv_results_)
        model_result.to_csv('result.csv')
    
        history = model.best_estimator_

        # 検証用セットを用いて評価
        print(model.score(x_val, y_val))
    elif opts.model_name == 'nn':
        model = nn_model()
        optimizer = optimizers.Adam(lr=opts.lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
        history = model.fit(x_train, y_train, batch_size=opts.batch_size, epochs=opts.epochs, verbose=1, validation_split=0.2)

    metrics = ['loss', 'accuracy']
    for i in range(len(metrics)):
        metric = metrics[i]
        plt.subplot(1, 2, i+1)
        plt.title(metric)

        plt_train = history.history[metric]
        plt_test = history.history['val_'+metric]

        plt.plot(plt_train, label = 'train')
        plt.plot(plt_test, label = 'validation')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    main()